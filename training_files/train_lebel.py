#!/usr/bin/env python3

import argparse
import logging
from typing import Dict, List, Union, Any
import numpy as np
from datetime import datetime

from encoding.assembly.assembly_generator import AssemblyGenerator
from encoding.features import LanguageModelFeatureExtractor, SpeechFeatureExtractor
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.utils import zs, ActivationCache, ModelSaver
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


class LebelTrainer:
    """A class to handle training and evaluation of encoding models on the Lebel dataset."""

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

    def setup_logger(self):
        """Initialize experiment logger (wandb or tensorboard)."""
        backend = self.config.get("logger_backend", "wandb").lower()
        if backend == "wandb":
            # Lazily import and initialize wandb
            try:
                import wandb  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "wandb selected as logger_backend but not installed. Install with: pip install wandb"
                ) from e
            project_name = self.config["wandb_project_name"]
            wandb.init(
                project=project_name,
                config=self.config,
                name=f"lebel-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            self.experiment_logger = WandBLogger()
        elif backend == "tensorboard":
            # Use timestamped run directory inside results_dir
            run_dir = f"{self.config.get('results_dir', 'results')}/runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.experiment_logger = TensorBoardLogger(log_dir=run_dir)
        else:
            raise ValueError(
                f"Unsupported logger_backend '{backend}'. Use 'wandb' or 'tensorboard'."
            )
        self.brain_plotter = BrainPlotter(self.experiment_logger)

    def setup_assembly(self):
        """Initialize the Lebel assembly."""
        self.assembly = AssemblyGenerator.generate_assembly(
            dataset_type="lebel",
            data_dir=self.config["data_dir"],
            subject=self.config["subject"],
            lookback=self.config["lookback"],
            context_type=self.config["context_type"],
        )
        self.logger.info(f"Assembly loaded with {len(self.assembly.stories)} stories")
        self.logger.info(f"Using context type: {self.config['context_type']}")
        if self.config["use_volume"]:
            self.logger.info("Using volume data")
        else:
            self.logger.info("Using surface data")

    def setup_models(self):
        """Initialize feature extractor and downsampler."""
        self.embeddings_model = LanguageModelFeatureExtractor(
            {
                "model_name": self.config["model_name"],
                "layer_idx": self.config["layer_idx"],
                "last_token": self.config["last_token"],
            }
        )
        self.downsampler = Downsampler()
        self.model = NestedCVModel(model_name="ridge_regression")

    def prepare_data(self) -> Dict[str, np.ndarray]:
        """Prepare training and test data with downsampling.

        Returns:
            Dictionary containing prepared data arrays
        """
        train_stories = self.assembly.stories[:-1]
        test_stories = self.assembly.stories[-1:]

        downsampled_X_train = {}
        downsampled_X_test = {}

        # Process each story
        for story in train_stories + test_stories:
            idx = self.assembly.stories.index(story)
            texts = self.assembly.get_stimuli()[idx]

            # Try to load cached activations with new multi-layer system
            cache_key = self.activation_cache._get_cache_key(
                story=story,
                lookback=self.config["lookback"],
                model_name=self.config["model_name"],
                context_type=self.config["context_type"],
                last_token=self.config["last_token"],
                dataset_type="lebel",
                raw=True,
            )

            # Try to load from cache first
            lazy_cache = self.activation_cache.load_multi_layer_activations(cache_key)

            if lazy_cache is not None:
                self.logger.info(f"Loaded cached activations for story {story}")
                # Validate context type
                try:
                    lazy_cache.validate_context_type(self.config["context_type"])
                except ValueError as e:
                    self.logger.warning(f"Context type mismatch for story {story}: {e}")
                    lazy_cache = None

            if lazy_cache is not None:
                # Load the specific layer we need
                features = lazy_cache.get_layer(self.config["layer_idx"])
            else:
                # If not in cache, compute all layers and cache them
                self.logger.info(f"Computing activations for story {story}")
                all_layer_features = self.embeddings_model.extract_all_layers(texts)

                # Create metadata for caching
                metadata = {
                    "model_name": self.config["model_name"],
                    "story": story,
                    "lookback": self.config["lookback"],
                    "context_type": self.config["context_type"],
                    "hook_type": self.embeddings_model.hook_type,
                    "last_token": self.config["last_token"],
                    "dataset_type": "lebel",
                    "available_layers": list(all_layer_features.keys()),
                    "created_at": datetime.now().isoformat(),
                    "raw": True,
                }

                # Save to cache
                self.activation_cache.save_multi_layer_activations(
                    cache_key, all_layer_features, metadata
                )

                # Get the specific layer we need
                features = all_layer_features[self.config["layer_idx"]]
            # features = self.assembly.get_word_rates()[idx]
            # features = np.array(features)
            # print(features.shape)

            # Get timing information
            split_indices = self.assembly.get_split_indices()[idx]
            data_times = self.assembly.get_data_times()[idx]
            tr_times = self.assembly.get_tr_times()[idx]

            # Downsample features
            downsampled_features = self.downsampler.downsample(
                data=features,
                data_times=data_times,
                tr_times=tr_times,
                method=self.config["downsample_method"],
                split_indices=(
                    split_indices
                    if "average" in self.config["downsample_method"]
                    or "sum" in self.config["downsample_method"]
                    or "last" in self.config["downsample_method"]
                    else None
                ),
                window=self.config["lanczos_window"],
                cutoff_mult=self.config["lanczos_cutoff_mult"],
            )

            if story in train_stories:
                downsampled_X_train[story] = downsampled_features
            else:
                downsampled_X_test[story] = downsampled_features
        # save downsampled_X_train and downsampled_X_test

        # Prepare final arrays
        trim_start = self.config["trim_start"]
        trim_end = self.config["trim_end"]

        Rstim = np.nan_to_num(
            np.vstack(
                [zs(downsampled_X_train[story][10:-5]) for story in train_stories]
            )
        )
        Pstim = np.nan_to_num(
            np.vstack(
                [
                    zs(downsampled_X_test[story][trim_start:-trim_end])
                    for story in test_stories
                ]
            )
        )

        # Create delayed features
        delays = range(1, self.config["ndelays"] + 1)
        Rstim = FIR.make_delayed(Rstim, delays)
        Pstim = FIR.make_delayed(Pstim, delays)

        # Get brain data for each story
        brain_data_list = self.assembly.get_brain_data()

        # Stack and normalize brain responses
        Rresp = np.vstack(
            [
                zs(brain_data_list[self.assembly.stories.index(story)])
                for story in train_stories
            ]
        )
        Presp = np.vstack(
            [
                zs(brain_data_list[self.assembly.stories.index(story)][40:])
                for story in test_stories
            ]
        )
        print(Rresp.shape, Presp.shape)
        print(Rstim.shape, Pstim.shape)

        return {
            "Rstim": Rstim,
            "Pstim": Pstim,
            "Rresp": Rresp,
            "Presp": Presp,
        }

    def train(self) -> Dict[str, Any]:
        """Run the training process."""
        try:
            # Prepare data
            data = self.prepare_data()

            # Run nested cross-validation
            metrics, weights, best_alphas = self.model.fit_predict(
                features=data["Rstim"],
                targets=data["Rresp"],
                X_test=data["Pstim"],
                y_test=data["Presp"],
                groups=self.assembly.get_coord("stimulus_id"),
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

            path = self.model_saver.save_encoding_model(
                weights=weights,
                best_alphas=best_alphas,
                hyperparams=hyperparams,
                metrics=metrics,
            )
            self.logger.info(f"Model saved to {path}")

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
        language_mask = np.load("/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/lh_rh_lana_atlas_fsavg5_top_10pct_mask.npy")  # replace with your own path
        a1_mask = np.load("/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/glasser_a1_fsavg5_mask_bool_fixed.npy")  # replace with your own path
        v1_mask = np.load("/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/glasser_v1_fsavg5_mask_bool_fixed.npy")  # replace with your own path

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
        description="Train encoding model on Lebel dataset"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the Lebel dataset directory",
    )
    parser.add_argument(
        "--subject", type=str, default="UTS03", help="Subject ID to use"
    )
    parser.add_argument(
        "--context_type",
        type=str,
        default="fullcontext",
        choices=["fullcontext", "nocontext", "halfcontext"],
        help="Type of context window to use for processing stimuli",
    )
    parser.add_argument(
        "--use_volume",
        action="store_true",
        help="Use volume data instead of surface data",
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
        default="average",
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
        "--trim_start",
        type=int,
        default=50,
        help="Number of samples to trim from start",
    )
    parser.add_argument(
        "--trim_end", type=int, default=5, help="Number of samples to trim from end"
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
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to store cached activations",
    )

    # Delay parameters
    parser.add_argument(
        "--ndelays",
        type=int,
        default=4,
        help="Number of FIR delays to use (default: 4, representing 2s, 4s, 6s, 8s) TR dependent",
    )

    # Hardware parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")

    # New argument for results directory
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

    # Weights & Biases parameters (used if using wandb)
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
    trainer = LebelTrainer(config)

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
