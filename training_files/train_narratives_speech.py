#!/usr/bin/env python3
"""
Train encoding model on Narratives dataset using speech model features.
"""
import argparse
import logging
from typing import Dict, Any
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.utils import ModelSaver, SpeechActivationCache
from encoding.features.FIR_expander import FIR
from encoding.assembly.assembly_generator import AssemblyGenerator
from encoding.features import SpeechFeatureExtractor
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


class NarrativesSpeechTrainer:
    """Trainer for encoding models using speech model features."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_logger()
        self.model_saver = ModelSaver(
            base_dir=self.config.get("results_dir", "results")
        )
        speech_extractor = SpeechFeatureExtractor(
            model_name=self.config["speech_model_name"],
            chunk_size=0.1,
            context_size=16.0,
            layer=self.config["layer_idx"],
            pool="last",
            target_sample_rate=16000,
            device="cpu",
            disable_tqdm=False,
        )
        self.embeddings_model = speech_extractor
        self.assembly = None
        self.speech_cache = SpeechActivationCache(cache_dir=self.config["cache_dir"])
        self.downsampler = Downsampler()
        self.model = NestedCVModel(model_name="ridge_regression")
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
                name=f"narratives-speech-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
        """Initialize the Narratives assembly (loads all stories for the subject)."""
        mask_path = self.config.get("mask_path", None)
        self.assembly = AssemblyGenerator.generate_assembly(
            dataset_type="narratives",
            data_dir=self.config["data_dir"],
            subject=self.config["subject"],
            tr=self.config["tr"],
            lookback=self.config["lookback"],
            use_volume=self.config.get("use_volume", False),
            mask_path=mask_path,
        )
        self.logger.info(f"Assembly loaded with {len(self.assembly.stories)} stories")

    def prepare_data(self) -> Dict[str, np.ndarray]:
        """
        Extract features from wav, align to TRs, and prepare X, Y for training.
        - Features and their times are extracted from the speech model (not the assembly).
        - TR times and brain data are always from the assembly.
        - Downsampling aligns speech features (using their own times) to the assembly's TR times.
        """
        # Select story
        story = self.config.get("story", "21styear")
        story_idx = self.assembly.stories.index(story)
        # Get wav path for the story
        # could also be a path to a wav file

        wav_path = self.assembly.get_audio_path()[story_idx]

        cache_key = self.speech_cache.get_cache_key(
            audio_id="/Users/tahabinhuraib/Documents/lit_encoding/data/narratives/neural_data/sub-249/21styear_audio_trimmed.wav",  # or a content hash if paths can move
            model_name=self.embeddings_model.model_name,
            chunk_size=self.embeddings_model.chunk_size,
            context_size=self.embeddings_model.context_size,
            pool=self.embeddings_model.pool,
            target_sample_rate=self.embeddings_model.target_sample_rate,
            dataset_type="narratives",
            extra={"layer_mode": "all"},
        )
        lazy = self.speech_cache.load_multi_layer_activations(cache_key)

        if lazy is not None:
            # sanity-check the important knobs match
            lazy.validate_params(
                expected={
                    "model_name": self.embeddings_model.model_name,
                    "chunk_size": self.embeddings_model.chunk_size,
                    "context_size": self.embeddings_model.context_size,
                    "pool": self.embeddings_model.pool,
                    "target_sample_rate": self.embeddings_model.target_sample_rate,
                    "dataset_type": "narratives",
                }
            )

            # resolve layer index if you asked for "last"

            features = lazy.get_layer(self.config["layer_idx"])  # [n_chunks, D]
            times = lazy.get_times()  # [n_chunks]
            if times is None:
                raise RuntimeError("Cached speech activations missing 'times' array.")

        else:
            # Not cached → compute all layers once, then cache
            layer_to_feats, times = self.embeddings_model.extract_all_layers(wav_path)
            if len(layer_to_feats) == 0:
                raise RuntimeError(
                    "extract_all_layers returned no layers (audio too short?)."
                )

            # metadata saved alongside for future validation
            metadata = {
                "modality": "speech",
                "audio_id": "/Users/tahabinhuraib/Documents/lit_encoding/data/narratives/neural_data/sub-249/21styear_audio_trimmed.wav",  # replace with your own path
                "model_name": self.embeddings_model.model_name,
                "chunk_size": self.embeddings_model.chunk_size,
                "context_size": self.embeddings_model.context_size,
                "pool": self.embeddings_model.pool,
                "target_sample_rate": self.embeddings_model.target_sample_rate,
                "dataset_type": "narratives",
                "available_layers": sorted(layer_to_feats.keys()),
            }

            self.speech_cache.save_multi_layer_activations(
                cache_key,
                all_layer_activations=layer_to_feats,
                metadata=metadata,
                times=times,
            )

            # resolve layer index if you asked for "last"

            features = layer_to_feats[self.config["layer_idx"]]  # [n_chunks, D]

        # Get TR times and brain data from the assembly
        tr_times = self.assembly.get_tr_times()[story_idx]
        brain_data = self.assembly.get_brain_data()[story_idx]
        split_indices = self.assembly.get_split_indices()[story_idx]
        # Downsample speech features to TRs using the Downsampler (method='lanczos')
        downsampled_features = self.downsampler.downsample(
            data=features,
            data_times=times,
            tr_times=tr_times,
            method="legacy_average",
            window=self.config.get("lanczos_window", 3),
            cutoff_mult=self.config.get("lanczos_cutoff_mult", 1.0),
            split_indices=split_indices,
        )
        # Apply FIR delays (make_delayed)
        ndelays = self.config.get("ndelays", 8)
        delays = range(1, ndelays + 1)
        delayed_features = FIR.make_delayed(downsampled_features, delays)
        # Trim X and Y exactly as in train_narratives.py
        X = delayed_features[14:-9]
        Y = brain_data[14:-9]
        self.logger.info(f"X shape after FIR/delays and trimming: {X.shape}")
        self.logger.info(f"Y shape after trimming: {Y.shape}")
        return {"X": X, "Y": Y}

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to the configured backend and plot brain results."""
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
        language_mask = np.load(
            "/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/lh_rh_lana_atlas_fsavg5_top_10pct_mask.npy"
        )
        a1_mask = np.load(
            "/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/glasser_a1_fsavg5_mask_bool_fixed.npy"
        )
        v1_mask = np.load(
            "/storage/coda1/p-aivanova7/0/shared/litcoder_release/litcoder_release/masks/glasser_v1_fsavg5_mask_bool_fixed.npy"
        )

        # Log brain plots if available
        if "correlations" in metrics and "significant_mask" in metrics:
            correlations = np.array(metrics["correlations"])
            significant_mask = np.array(metrics["significant_mask"], dtype=bool)
            self.brain_plotter.log_plots(
                correlations=correlations,
                significant_mask=significant_mask,
                prefix="",
                is_volume=self.config.get("use_volume", False),
                language_mask=language_mask,
                roi_masks={"a1": a1_mask, "v1": v1_mask},
            )
        if "best_alpha" in metrics:
            self.experiment_logger.log_scalar(
                "best_alpha", float(metrics["best_alpha"])
            )
        if "n_significant" in metrics:
            self.experiment_logger.log_scalar(
                "n_significant_voxels", float(metrics["n_significant"])
            )

    def train(self) -> Dict[str, Any]:
        data = self.prepare_data()
        metrics, weights, best_alphas = self.model.fit_predict(
            features=data["X"],
            targets=data["Y"],
            folding_type=self.config["folding_type"],
            n_outer_folds=self.config["n_outer_folds"],
            n_inner_folds=self.config["n_inner_folds"],
            chunk_length=self.config["chunk_length"],
            singcutoff=self.config["singcutoff"],
            use_gpu=self.config["use_gpu"],
            single_alpha=True,
            normalpha=True,
            use_corr=True,
            normalize_features=self.config["normalize_features"],
            normalize_targets=self.config["normalize_targets"],
        )
        # Save model
        hyperparams = {
            "folding_type": self.config["folding_type"],
            "n_outer_folds": self.config["n_outer_folds"],
            "n_inner_folds": self.config["n_inner_folds"],
            "chunk_length": self.config["chunk_length"],
            "singcutoff": self.config["singcutoff"],
            "use_gpu": self.config["use_gpu"],
            "single_alpha": True,
            "normalpha": True,
            "use_corr": True,
            "normalize_features": self.config["normalize_features"],
            "normalize_targets": self.config["normalize_targets"],
            "features": "speech",
        }
        self.model_saver.save_encoding_model(
            weights=weights,
            best_alphas=best_alphas,
            hyperparams=hyperparams,
            metrics=metrics,
        )
        self.log_metrics(metrics)
        self.logger.info(f"Median correlation: {metrics['median_score']:.3f}")
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train encoding model on Narratives dataset using speech features"
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
    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-tiny",
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
    # Speech model parameters
    parser.add_argument(
        "--wav_path", type=str, required=False, help="Path to wav file for the story"
    )
    parser.add_argument(
        "--speech_model_name",
        type=str,
        default="openai/whisper-tiny",
        help="HuggingFace speech model name",
    )
    parser.add_argument(
        "--chunk_size", type=float, default=0.1, help="Chunk size in seconds"
    )
    parser.add_argument(
        "--context_size", type=float, default=16.0, help="Context size in seconds"
    )
    # Downsampling parameters
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
        "--lookback",
        type=int,
        default=256,
        help="Number of words to look back for context",
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
        "--normalize_features", action="store_true", help="Normalize input features"
    )
    parser.add_argument(
        "--normalize_targets", action="store_true", help="Normalize target values"
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
        default="cache_narratives_speech",
        help="Directory to store cached activations",
    )
    # Delay parameters
    parser.add_argument(
        "--ndelays",
        type=int,
        default=8,
        help="Number of FIR delays to use (default: 8)",
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
    args = parse_args()
    config = vars(args)
    trainer = NarrativesSpeechTrainer(config)
    trainer.setup_assembly()
    metrics = trainer.train()
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
