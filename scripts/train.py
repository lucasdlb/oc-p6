"""Training script entry point."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from credit_risk.config.paths import MODELS_DIR
from credit_risk.config.settings import load_settings
from credit_risk.pipeline.training_pipeline import TrainingPipeline
from credit_risk.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--experiment", type=str, default="credit_risk", help="MLflow experiment name"
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    logger = setup_logging(settings.logging)
    logger.info("Starting training pipeline")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = TrainingPipeline(settings)
    result = pipeline.run()

    logger.info(f"Training complete. Metrics: {result['metrics']}")
    logger.info(f"Model trained with {result['train_shape'][1]} features")

    model_path = MODELS_DIR / "model.pkl"
    pipeline.trainer.save(str(model_path))
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
