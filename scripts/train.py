"""Training script entry point."""

import argparse
from pathlib import Path

from credit_risk.pipeline.training_pipeline import TrainingPipeline
from credit_risk.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument(
        "--experiment", type=str, default="credit_risk", help="MLflow experiment name"
    )
    _ = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting training pipeline")

    models_path = Path("models")
    models_path.mkdir(parents=True, exist_ok=True)

    pipeline = TrainingPipeline()
    result = pipeline.run()

    logger.info(f"Training complete. Metrics: {result['metrics']}")
    logger.info(f"Model trained with {result['train_shape'][1]} features")

    model_path = models_path / "model.pkl"
    pipeline.trainer.save(str(model_path))
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
