import json
import logging
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the ML pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "training.log"

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging(True)


# Data Loading and Preprocessing
def load_data(filepath: Path) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def prepare_train_test_data(
    data: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare training and testing datasets with proper encoding."""
    # Separate features and target
    y = data["target"]
    X = data.drop(columns=["target", "item_id", "user_id"]).astype(float)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Encode target variables
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train, X_test, y_train_encoded, y_test_encoded


def validate_data_path(filepath: Path) -> bool:
    """Validate that data file exists."""
    if not filepath.exists():
        print(f"Data file does not exist: {filepath}")
        return False
    return True


# Model Utilities
def extract_non_default_params(model: BaseEstimator) -> Dict:
    """Extract model parameters that differ from defaults."""
    sig = signature(model.__class__.__init__)
    defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not v.empty
    }
    current_params = model.get_params()
    return {
        k: v for k, v in current_params.items() if k in defaults and v != defaults[k]
    }


def curate_model_name(title: str) -> str:
    """Generate model filename with timestamp and proper extension."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{title.lower()}_{timestamp}"
    return name


def save_model_artifacts(
    model: ClassifierMixin,
    metadata: Dict,
    classifier_title: str,
    report_df: pd.DataFrame,
    filepath: str,
) -> Tuple[Path, Path, Path]:
    """Save model, metadata, and classification report."""
    model_dir = Path(filepath)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / f"{classifier_title}.pkl"
    joblib.dump(model, model_path)

    # Create metadata path
    meta_path = model_dir / f"metadata_{classifier_title}.json"

    # Save classification report
    report_path = model_dir / f"report_{classifier_title}.csv"
    report_df.to_csv(report_path, index=False)

    print(f"Saved {classifier_title} artifacts to: {model_dir}")
    return model_path, meta_path, report_path


def save_feature_importance_plot(
    classifier: ClassifierMixin,
    X_train: pd.DataFrame,
    filepath: str,
    top_n: int = 20,
) -> pd.Series:
    """Save feature importance plot if model supports it."""
    if not hasattr(classifier, "feature_importances_"):
        return pd.Series()

    feature_names = (
        X_train.columns
        if isinstance(X_train, pd.DataFrame)
        else np.arange(X_train.shape[1])
    )
    importances = classifier.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # Create and save plot
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    feat_imp.head(top_n).plot(kind="barh", title=f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(filepath, dpi=250, bbox_inches="tight")
    plt.close()

    return feat_imp


def pascal_to_snake(title: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", title).lower()


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
            "version": self.version,
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
        logger.info(f"\n{classifier_title.upper()} Results:")
        logger.info(f"CV F1 Weighted: {metrics['cv_f1_weighted']}")
        logger.info(f"Test Accuracy: {metrics['accuracy']['test']}")
        logger.info(f"Test F1 Weighted: {metrics['f1_weighted']['test']}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']['test']}")

        # Print detailed classification report
        print(f"\n{classifier_title.upper()} Classification Report:")
        logger.info(classification_report(self.y_test, test_pred))


# Model Configuration
def get_model_configs(random_state: int = 42) -> Dict[str, ClassifierMixin]:
    """Get configured models for training."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        ),
    }


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
        logger.info(f"\nTraining {name}...")
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
    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        default="single",
        help="Training execution mode: 'single' for one model, 'all' for all models",
    )

    # Model selection (only relevant for single mode)
    parser.add_argument(
        "--model",
        choices=[
            "logistic_regression",
            "xgboost",
            "random_forest",
            "lightgbm",
            "neural_network",
        ],
        default="logistic_regression",
        help="Model to train (only used when mode='single')",
    )

    # Add data path as argument
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/training/train.csv"),
        help="Path to training data file",
    )
    # Add model path as argument
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/classifier"),
        help="Path to model output files",
    )

    return parser.parse_args()


def validate_arguments(args: Namespace) -> None:
    """Validate argument combinations and warn about unused options."""
    if args.mode == "all":
        logger.warning(f"--model argument '{args.model}' ignored when mode='all'")


def save_metadata(results: dict, model_name: str, debug: bool = False) -> None:
    """Save model metadata to JSON file with timestamp."""

    first_result = next(iter(results.values()))

    if debug:
        print(f"Model path: {first_result['paths']['model']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(first_result["paths"]["model"]).parent
    filepath = model_dir / f"metadata_{model_name.lower()}_{timestamp}.json"

    if debug:
        print(f"Saving metadata to: {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main() -> None:
    """Main ML pipeline execution."""
    # Parse CLI arguments first
    args = parse_cli_arguments()

    # Validate argument combinations
    validate_arguments(args)

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
        if args.mode == "all":
            logger.info("Training all models...")
            results = train_multiple_models(X_train, X_test, y_train, y_test, len(data))
        elif args.mode == "single":
            logger.info(f"Training single model: {args.model}")
            results = train_model(
                args.model, X_train, X_test, y_train, y_test, len(data), args.out
            )
            # save_metadata(results, args.model)
            save_metadata(results, args.model, debug=True)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}. Options: [single, all]")

        logger.info("ML pipeline completed successfully!")

        # log results summary
        # print(json.dumps(results, indent=3))
        if args.verbose and results:
            logger.debug(f"Training results: {results}")

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
