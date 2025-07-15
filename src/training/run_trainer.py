import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

from mlflow import MlflowClient
import dvc.api
import mlflow
from mlflow.models import infer_signature
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
    path: Path
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
        feat_imp_title = self.path / f"feat_plot_{classifier_title}.png"
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
            self.classifier, classifier_title, report_df, self.path
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

        return self.classifier, enriched_metadata


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
        model, results[name] = trainer.train_and_evaluate()

    return model, results


def train_model(
    model_title: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    data_size: int,
    filepath: Path,
) -> Dict:
    """Train single model and return results."""
    model = get_model_configs().get(model_title)
    results = {}
    logger.info(f"\nTraining {model_title}...")
    trainer = ModelTrainer(model, X_train, X_test, y_train, y_test, data_size, filepath)
    model, results[model_title] = trainer.train_and_evaluate()

    # TODO: Add Mlflow here ...

    return model, results


def parse_cli_arguments() -> Namespace:
    """Parse command-line arguments for model training execution."""
    parser = ArgumentParser(description="Model training CLI")

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


def create_exp(exp_name: str = "RecoSysML", port: str = "8083") -> None:
    """Create an MLflow experiment for the recommendation system."""
    print("Creatig Mlflow experiment: ", exp_name)
    client = MlflowClient(tracking_uri=f"http://127.0.0.1:{port}")

    description = "This is the recommendation system to personalise user's offers."

    exp_tags = {
        "project_name": "personalized recommendation system",
        "mlflow.note.content": description,
    }

    def find_experiment(exp_name: str = "Default") -> bool:
        """Check if experiment exists."""
        all_experiments = client.search_experiments()

        for experiment in all_experiments:
            if experiment.name == exp_name:
                return True

        return False

    # Check if experiment already exists
    if find_experiment(exp_name):
        print(f"Experiment '{exp_name}' already exists!")
    else:
        # Create the experiment
        artifact_location = Path.cwd().joinpath("mlruns").as_uri()
        experiment_id = client.create_experiment(
            name=exp_name, tags=exp_tags, artifact_location=artifact_location
        )

        print(f"Created experiment '{exp_name}' with ID: {experiment_id}")


def flatten_metrics(results: dict) -> dict:
    """Flatten nested metrics dictionary for MLflow logging."""
    sub_results = next(iter(results.values()))
    metadata = sub_results["metadata"]
    metrics = {}
    metrics["data-size"] = float(metadata["data_size"])

    for name, value in metadata["metrics"].items():
        if isinstance(value, dict):
            train, test = list(value.values())
            metrics[f"train.{name}"] = train
            metrics[f"test.{name}"] = test
        else:
            metrics[name] = value

    return metrics


def main() -> None:
    """Main ML pipeline execution."""

    # Load DVC parameters
    params = dvc.api.params_show()
    model = params["train"]["model"]
    mode = params["train"]["mode"]

    # Parse CLI arguments
    args = parse_cli_arguments()
    model_path = Path(args.output)
    model_path.mkdir(parents=True, exist_ok=True)

    # MLFlow Tracking setup
    port = "8084"
    exp_name = "RecoSysML"
    create_exp(exp_name, port)
    mlflow.set_tracking_uri(f"http://127.0.0.1:{port}")
    mlflow.set_experiment(exp_name)

    try:
        # Setup and validate data path
        data_path = args.data_path
        logger.info(f"Using data path: {data_path}")

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
            trained_model, results = train_multiple_models(
                X_train, X_test, y_train, y_test, len(data)
            )

        elif mode.strip() == "single":
            logger.info(f"Training single model: {model}")
            trained_model, results = train_model(
                model, X_train, X_test, y_train, y_test, len(data), model_path
            )

            # Saving trained model artifacts on local machine
            save_metadata(results, model, debug=True)

            # MLflow experiment logging
            model_title = "".join(list(results.keys()))
            artifact_path = f"{model_title}_{exp_name.lower()}"

            metrics = flatten_metrics(results)
            sub_results = next(iter(results.values()))
            params = sub_results["metadata"]["params"]
            report_path = Path(sub_results["paths"]["report"])
            data_size = float(sub_results["metadata"]["data_size"])
            signature = infer_signature(X_train, trained_model.predict(X_train))

            with mlflow.start_run(run_name=model_title) as _:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_metric("data-size", data_size)
                mlflow.log_artifact(report_path, "classification-report")

                mlflow.sklearn.log_model(
                    sk_model=trained_model,
                    input_example=X_train[:5],
                    signature=signature,
                    name=artifact_path,
                )

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
