#!/usr/bin/env python3

import argparse
import logging
from typing import Dict, List, Union, Any
import numpy as np

from datetime import datetime

from encoding.assembly.assembly_generator import AssemblyGenerator
from encoding.features import LanguageModelFeatureExtractor, WordRateFeatureExtractor
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.utils import ActivationCache, ModelSaver
from encoding.features.FIR_expander import FIR
from encoding.plotting.plotting_utils import (
    BrainPlotter,
    TensorBoardLogger,
    WandBLogger,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NarrativesTrainer:
    """A class to handle training and evaluation of encoding models on the Narratives dataset."""

    def __init__(self, config: Dict):
        """Initialize the trainer with configuration parameters.

        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_logger()
        self.setup_assembly()
        self.setup_models()
        self.activation_cache = ActivationCache(cache_dir=self.config["cache_dir"])
        self.model_saver = ModelSaver(
            base_dir=self.config.get("results_dir", "results")
        )
        self.brain_plotter = BrainPlotter(self.experiment_logger)

    def setup_logger(self):
        """Initialize experiment logger (wandb or tensorboard)."""
        backend = self.config.get("logger_backend", "wandb").lower()
        if backend == "wandb":
            try:
                import wandb  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "wandb selected as logger_backend but not installed. Install with: pip install wandb"
                ) from e
            project_name = self.config.get("wandb_project_name", "lit-encoding")
            wandb.init(
                project=project_name,
                config=self.config,
                name=f"narratives-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            self.experiment_logger = WandBLogger()
        elif backend == "tensorboard":
            run_dir = f"{self.config.get('results_dir', 'results')}/runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.experiment_logger = TensorBoardLogger(log_dir=run_dir)
        else:
            raise ValueError(
                f"Unsupported logger_backend '{backend}'. Use 'wandb' or 'tensorboard'."
            )

    def setup_assembly(self):
        """Initialize the Narratives assembly."""
        print("this is the data dir: ", self.config["data_dir"])
        print("this is the subject: ", self.config["subject"])
        self.assembly = AssemblyGenerator.generate_assembly(
            dataset_type="narratives",  # This is dangerous: TODO: define this somewhere else.
            data_dir=self.config["data_dir"],
            subject=self.config["subject"],
            tr=self.config["tr"],
            lookback=self.config["lookback"],
            context_type=self.config["context_type"],
            use_volume=self.config["use_volume"],
        )
        self.logger.info(f"Assembly loaded with {len(self.assembly.stories)} stories")
        self.logger.info(f"Using context type: {self.config['context_type']}")
        if self.config["use_volume"]:
            self.logger.info("Using volume data")
        else:
            self.logger.info("Using surface data")

    def setup_models(self):
        """Initialize feature extractor and downsampler."""
        self.word_rate_extractor = WordRateFeatureExtractor({})
        self.model = NestedCVModel(model_name="ridge_regression")

    def prepare_data(self) -> Dict[str, np.ndarray]:
        """Prepare training and test data with downsampling.

        Returns:
            Dictionary containing prepared data arrays
        """
        downsampled_X = {}
        brain_data = {}

        # Process each story
        for story in self.assembly.stories:
            idx = self.assembly.stories.index(story)
            texts = self.assembly.get_stimuli()[idx]
            word_rates = self.assembly.get_word_rates()[idx]
            features = self.word_rate_extractor.extract_features(word_rates)

            # Get timing information
            try:
                print(f"this is the shape of the features: {features.shape}")
            except Exception as e:
                print(f"this is the error: {e}")

            # Downsample features
            downsampled_X[story] = features
            brain_data[story] = self.assembly.get_brain_data()[idx]

        # Create delayed features
        delays = range(1, self.config["ndelays"] + 1)
        delayed_features = {}
        for story in self.assembly.stories:
            delayed_features[story] = FIR.make_delayed(downsampled_X[story], delays)

        story_order = ["21styear"]
        X = np.concatenate([delayed_features[story] for story in story_order], axis=0)
        Y = np.concatenate([brain_data[story] for story in story_order], axis=0)
        print(f"this is the shape of X: {X.shape}")
        print(f"this is the shape of Y: {Y.shape}")
        # trim so that it becomes: 14:-9
        X = X[14:-9]
        Y = Y[14:-9]
        print(f"this is the shape of X after trimming: {X.shape}")
        print(f"this is the shape of Y after trimming: {Y.shape}")

        return {
            "X": X,
            "Y": Y,
        }

    def train(self) -> Dict[str, Any]:
        """Run the training process."""
        try:
            # Prepare data
            data = self.prepare_data()
            # Run nested cross-validation
            metrics, weights, best_alphas = self.model.fit_predict(
                features=data["X"],
                targets=data["Y"],
                folding_type=self.config["folding_type"],
                n_outer_folds=self.config["n_outer_folds"],
                n_inner_folds=self.config["n_inner_folds"],
                chunk_length=self.config["chunk_length"],
                singcutoff=self.config["singcutoff"],
                use_gpu=self.config["use_gpu"],
                single_alpha=True,  # Always use single alpha
                normalpha=True,  # Always normalize alpha
                use_corr=True,  # Always use correlation
                normalize_features=self.config["normalize_features"],
                normalize_targets=self.config["normalize_targets"],
            )

            # Log metrics to configured backend
            self.log_metrics(metrics)

            # Save model weights and hyperparameters
            hyperparams = {
                # Copy ALL configuration parameters
                **self.config,
                # Add hardcoded parameters that are always set
                "single_alpha": True,  # Always use single alpha
                "normalpha": True,  # Always normalize alpha
                "use_corr": True,  # Always use correlation
            }

            path_with_metrics = self.model_saver.save_encoding_model(
                weights=weights,
                best_alphas=best_alphas,
                hyperparams=hyperparams,
                metrics=metrics,
            )
            print(f"this is the path with metrics: {path_with_metrics}")

            # Log metrics
            self.logger.info("\nTraining Results:")
            self.logger.info(f"Median correlation: {metrics['median_score']:.3f}")
            self.logger.info(
                f"Significant voxels: {metrics['n_significant']}/{len(metrics['correlations'])} ({metrics['percent_significant']:.1f}%)"
            )

            if "median_significant_score" in metrics:
                self.logger.info(
                    f"Median correlation (significant voxels): {metrics['median_significant_score']:.3f}"
                )

            return metrics

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def log_metrics(self, metrics: Dict[str, Union[float, List[float]]]):
        """Log metrics to the configured backend.

        Args:
            metrics: Dictionary containing training metrics
        """
        # Scalar summaries
        self.experiment_logger.log_scalar(
            "median_correlation", float(metrics["median_score"])
        )
        self.experiment_logger.log_scalar(
            "mean_correlation", float(metrics["mean_score"])
        )
        self.experiment_logger.log_scalar(
            "std_correlation", float(metrics["std_score"])
        )
        self.experiment_logger.log_scalar(
            "min_correlation", float(metrics["min_score"])
        )
        self.experiment_logger.log_scalar(
            "max_correlation", float(metrics["max_score"])
        )
        language_mask = np.load("/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/lh_rh_lana_atlas_fsavg5_top_10pct_mask.npy")
        a1_mask = np.load("/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/glasser_a1_fsavg5_mask_bool_fixed.npy")
        v1_mask = np.load("/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/glasser_v1_fsavg5_mask_bool_fixed.npy")

        # Create and log brain surface plots and correlation histogram
        if "correlations" in metrics and "significant_mask" in metrics:
            correlations = np.array(metrics["correlations"])
            significant_mask = np.array(metrics["significant_mask"], dtype=bool)

            # Log all plots via BrainPlotter
            self.brain_plotter.log_plots(
                correlations=correlations,
                significant_mask=significant_mask,
                prefix="",
                is_volume=self.config["use_volume"],
                language_mask=language_mask,
                roi_masks={"a1": a1_mask, "v1": v1_mask},
            )

        # Log best alpha if available
        if "best_alpha" in metrics:
            self.experiment_logger.log_scalar(
                "best_alpha", float(metrics["best_alpha"])
            )

        # Log number of significant voxels if available
        if "n_significant" in metrics:
            self.experiment_logger.log_scalar(
                "n_significant_voxels", float(metrics["n_significant"])
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train encoding model on Narratives dataset"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the Narratives dataset directory",
    )
    parser.add_argument(
        "--subject", type=str, default="sub-59", help="Subject ID to use"
    )
    parser.add_argument(
        "--tr", type=float, default=1.5, help="TR value for the dataset"
    )
    parser.add_argument(
        "--context_type",
        type=str,
        default="fullcontext",
        choices=["fullcontext", "nocontext", "halfcontext"],
        help="Type of context window to use for processing stimuli",
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2-small",
        help="Name of the language model to use",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=9, help="Layer index to extract features from"
    )
    parser.add_argument(
        "--last_token",
        action="store_true",
        help="Whether to use only the last token's features",
    )

    # Training parameters
    parser.add_argument(
        "--n_outer_folds", type=int, default=5, help="Number of outer CV folds"
    )
    parser.add_argument(
        "--n_inner_folds", type=int, default=5, help="Number of inner CV folds"
    )
    parser.add_argument(
        "--folding_type",
        type=str,
        default="chunked",
        help="Type of cross-validation folding",
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=20,
        help="Length of chunks for chunked folding",
    )

    # Ridge regression parameters
    parser.add_argument(
        "--singcutoff",
        type=float,
        default=1e-10,
        help="Singular value cutoff for ridge regression (default: 1e-10)",
    )

    # Data preprocessing parameters
    parser.add_argument(
        "--downsample_method",
        type=str,
        default="lanczos",
        help="Method to use for downsampling (lanczos, sinc, average, etc.)",
    )
    parser.add_argument(
        "--lanczos_cutoff_mult",
        type=float,
        default=1.0,
        help="Cutoff multiplier for Lanczos filter",
    )
    parser.add_argument(
        "--lanczos_window", type=int, default=3, help="Window size for Lanczos filter"
    )
    parser.add_argument(
        "--normalize_features", action="store_true", help="Normalize input features"
    )
    parser.add_argument(
        "--normalize_targets", action="store_true", help="Normalize target values"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=256,
        help="Number of words to look back for context",
    )
    parser.add_argument(
        "--use_volume",
        action="store_true",
        help="Use volume data instead of surface data",
    )
    parser.add_argument(
        "--mask_path", type=str, default=None, help="Path to mask file for volume data"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_narratives",
        help="Directory to store cached activations",
    )

    # Delay parameters
    parser.add_argument(
        "--ndelays",
        type=int,
        default=8,
        help="Number of FIR delays to use (default: 6)",
    )

    # Hardware parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")

    # Results directory
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save model results",
    )

    # Logging backend selection
    parser.add_argument(
        "--logger_backend",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard"],
        help="Logging backend to use",
    )

    # Weights & Biases parameters
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        required=False,
        default="lit-encoding",
        help="Weights & Biases project name (used if --logger_backend wandb)",
    )

    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()

    # Convert args to dictionary
    config = vars(args)

    # Initialize trainer
    trainer = NarrativesTrainer(config)

    # Run training
    metrics = trainer.train()

    # Print final results
    logger.info("\n=== Final Results ===")
    logger.info(f"Median correlation: {metrics['median_score']:.4f}")
    logger.info(f"Mean correlation: {metrics['mean_score']:.4f}")
    logger.info(f"Std correlation: {metrics['std_score']:.4f}")
    logger.info(f"Min correlation: {metrics['min_score']:.4f}")
    logger.info(f"Max correlation: {metrics['max_score']:.4f}")

    if "best_alpha" in metrics:
        logger.info(f"Best alpha: {metrics['best_alpha']:.4f}")

    if "n_significant" in metrics:
        logger.info(f"Number of significant voxels: {metrics['n_significant']}")


if __name__ == "__main__":
    main()
