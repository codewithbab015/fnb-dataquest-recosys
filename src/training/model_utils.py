import logging
import re
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


logger = setup_logging(False)


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
    classifier_title: str,
    report_df: pd.DataFrame,
    filepath: Path,
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

    logger.info(f"Saved {classifier_title} artifacts to: {model_dir}")
    return model_path, meta_path, report_path


def save_feature_importance_plot(
    classifier: ClassifierMixin,
    X_train: pd.DataFrame,
    filepath: Path,
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
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        ),
    }
