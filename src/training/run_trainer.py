import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from model_utils import (
    curate_model_name,
    extract_non_default_params,
    get_model_configs,
    load_data,
    logger,
    pascal_to_snake,
    prepare_train_test_data,
    save_feature_importance_plot,
    save_model_artifacts,
    validate_data_path,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Model Training Class
@dataclass
class ModelTrainer:
    """Train and evaluate machine learning models."""

    classifier: Union[ClassifierMixin, BaseEstimator]
    X_train: Union[np.ndarray, pd.DataFrame]
    X_test: Union[np.ndarray, pd.DataFrame]
    y_train: Union[np.ndarray, pd.Series]
    y_test: Union[np.ndarray, pd.Series]
    data_size: int
    path: str
    decimal_places: int = 3

    def _calculate_metrics(
        self,
        train_pred: np.ndarray,
        test_pred: np.ndarray,
        train_proba: np.ndarray,
        test_proba: np.ndarray,
    ) -> Dict:
        """Calculate all evaluation metrics."""
        return {
            "accuracy": {
                "train": np.round(
                    accuracy_score(self.y_train, train_pred), self.decimal_places
                ),
                "test": np.round(
                    accuracy_score(self.y_test, test_pred), self.decimal_places
                ),
            },
            "f1_weighted": {
                "train": np.round(
                    f1_score(self.y_train, train_pred, average="weighted"),
                    self.decimal_places,
                ),
                "test": np.round(
                    f1_score(self.y_test, test_pred, average="weighted"),
                    self.decimal_places,
                ),
            },
            "roc_auc": {
                "train": np.round(
                    roc_auc_score(self.y_train, train_proba), self.decimal_places
                ),
                "test": np.round(
                    roc_auc_score(self.y_test, test_proba), self.decimal_places
                ),
            },
        }

    def _log_results(
        self, classifier_title: str, metrics: Dict, test_pred: np.ndarray
    ) -> None:
        """Log training results."""
        logger.info(f"\n{classifier_title.title()} Results:")
        logger.info(f"CV F1 Weighted: {metrics['cv_f1_weighted']}")
        logger.info(f"Test Accuracy: {metrics['accuracy']['test']}")
        logger.info(f"Test F1 Weighted: {metrics['f1_weighted']['test']}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']['test']}")

        # Print detailed classification report
        print(f"\n{classifier_title.upper()} Classification Report:")
        print(classification_report(self.y_test, test_pred))

    def train_and_evaluate(self) -> Dict:
        """Train model and return evaluation metrics."""
        title = pascal_to_snake(self.classifier.__class__.__name__)
        classifier_title = curate_model_name(title)
        filepath_output = self.path

        print(f"Training {classifier_title} model...")

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.classifier, self.X_train, self.y_train, cv=skf, scoring="f1_weighted"
        )

        # Train model
        self.classifier.fit(self.X_train, self.y_train)

        # Predictions
        train_pred = self.classifier.predict(self.X_train)
        test_pred = self.classifier.predict(self.X_test)
        train_proba = self.classifier.predict_proba(self.X_train)[:, 1]
        test_proba = self.classifier.predict_proba(self.X_test)[:, 1]

        # Metrics
        metrics = self._calculate_metrics(
            train_pred, test_pred, train_proba, test_proba
        )
        metrics["cv_f1_weighted"] = np.round(cv_scores.mean(), self.decimal_places)

        # Feature importance
        feat_imp_title = Path(filepath_output) / f"feat_plot_{classifier_title}.png"
        feat_imp = save_feature_importance_plot(
            self.classifier, self.X_train, feat_imp_title
        )

        # Classification report
        report_dict = classification_report(self.y_test, test_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        # Create metadata
        metadata = {
            "data_size": self.data_size,
            "classifier": classifier_title,
            "params": extract_non_default_params(self.classifier),
            "feature_importance": feat_imp.to_dict() if not feat_imp.empty else {},
            "metrics": metrics,
        }

        # Save artifacts
        model_path, meta_path, report_path = save_model_artifacts(
            self.classifier, metadata, classifier_title, report_df, filepath_output
        )

        enriched_metadata = {
            "metadata": metadata,
            "paths": {
                "model": str(model_path),
                "metadata": str(meta_path),
                "report": str(report_path),
            },
        }

        # Log results
        self._log_results(classifier_title, metrics, test_pred)

        return enriched_metadata


def train_multiple_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    data_size: int,
    version: float = 0.1,
) -> Dict:
    """Train multiple models and return results."""
    models = get_model_configs()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        trainer = ModelTrainer(
            model, X_train, X_test, y_train, y_test, data_size, version
        )
        results[name] = trainer.train_and_evaluate()

    return results


def train_model(
    model_title: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    data_size: int,
    filepath: str,
) -> Dict:
    """Train single model and return results."""
    model = get_model_configs().get(model_title)
    results = {}
    logger.info(f"\nTraining {model_title}...")
    trainer = ModelTrainer(model, X_train, X_test, y_train, y_test, data_size, filepath)
    results[model_title] = trainer.train_and_evaluate()

    return results


def parse_cli_arguments() -> Namespace:
    """Parse command-line arguments for model training execution."""
    parser = ArgumentParser(description="Model training CLI")

    # Training mode
    # parser.add_argument(
    #     "--mode",
    #     choices=["single", "all"],
    #     default="single",
    #     help="Training execution mode: 'single' for one model, 'all' for all models",
    # )

    # Model selection (only relevant for single mode)
    # parser.add_argument(
    #     "--model",
    #     choices=["logistic_regression", "xgboost", "random_forest", "lightgbm"],
    #     default="logistic_regression",
    #     help="Model to train (only used when mode='single')",
    # )

    # Add data path as argument
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/training/train.csv"),
        help="Path to training data file",
    )
    # Add model path as argument
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/classifier"),
        help="Path to model output files",
    )

    return parser.parse_args()


# def validate_arguments(mode: str, model: str) -> None:
#     """Validate argument combinations and warn about unused options."""
#     if mode == "all":
#         logger.warning(f"--model argument '{model}' ignored when mode='all'")


def save_metadata(results: dict, model_name: str, debug: bool = False) -> None:
    """Save model metadata to JSON file with timestamp."""

    first_result = next(iter(results.values()))

    if debug:
        logger.info(f"Model path: {first_result['paths']['model']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(first_result["paths"]["model"]).parent
    filepath = model_dir / f"metadata_{model_name.lower()}_{timestamp}.json"

    if debug:
        print(f"Saving metadata to: {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main() -> None:
    """Main ML pipeline execution."""
    import dvc.api

    params = dvc.api.params_show()
    print(params)

    model = params["train"]["model"]
    mode = params["train"]["mode"]
    # print(model, mode)
    # Parse CLI arguments first
    args = parse_cli_arguments()

    # # Validate argument combinations
    # validate_arguments(args)

    try:
        # Setup data path
        data_path = args.data_path
        logger.info(f"Using data path: {data_path}")

        # Validate data exists
        if not validate_data_path(data_path):
            logger.error(f"Data validation failed for path: {data_path}")
            return

        # Load and process data
        logger.info(f"Loading data from: {data_path}")
        data = load_data(data_path)
        logger.info(f"Data shape: {data.shape}")

        # Prepare train/test sets
        X_train, X_test, y_train, y_test = prepare_train_test_data(data)
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")

        # Train models based on mode
        if mode.strip() == "all":
            logger.info("Training all models...")
            results = train_multiple_models(X_train, X_test, y_train, y_test, len(data))
        elif mode.strip() == "single":
            logger.info(f"Training single model: {model}")
            results = train_model(
                model, X_train, X_test, y_train, y_test, len(data), args.output
            )
            # save_metadata(results, model)
            save_metadata(results, model, debug=True)
        else:
            raise ValueError(f"Unsupported mode: '{mode}'. Options: [single, all]")

        logger.info("ML pipeline completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid parameter: {e}")
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
