import json
import logging
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def setup_logging() -> logging.Logger:
    """Configure logging for the ML pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "ml_pipeline.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Data Loading and Preprocessing
def load_data(filepath: Path) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def prepare_train_test_data(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare training and testing datasets with proper encoding."""
    # Separate features and target
    y = data['target']
    X = data.drop(columns=['target', 'item_id', 'user_id']).astype(float)
    
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
        logger.error(f"Data file does not exist: {filepath}")
        return False
    return True

# Model Utilities
def extract_non_default_params(model: BaseEstimator) -> Dict:
    """Extract model parameters that differ from defaults."""
    sig = signature(model.__class__.__init__)
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not v.empty}
    current_params = model.get_params()
    return {k: v for k, v in current_params.items() if k in defaults and v != defaults[k]}

def save_model_artifacts(model: ClassifierMixin, metadata: Dict, classifier_name: str, 
                        version: float, report_df: pd.DataFrame) -> Tuple[Path, Path, Path]:
    """Save model, metadata, and classification report."""
    model_dir = Path("models/classifiers")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / f"{classifier_name}_model_v{version}.pkl"
    joblib.dump(model, model_path)
    
    # Save metadata
    meta_path = model_dir / f"{classifier_name}_metadata_v{version}.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    
    # Save classification report
    report_path = model_dir / f"{classifier_name}_report_v{version}.csv"
    report_df.to_csv(report_path, index=False)
    
    logger.info(f"Saved {classifier_name} artifacts to: {model_dir}")
    return model_path, meta_path, report_path

def save_feature_importance_plot(classifier: ClassifierMixin, X_train: pd.DataFrame, 
                               classifier_name: str, version: float, top_n: int = 20) -> pd.Series:
    """Save feature importance plot if model supports it."""
    if not hasattr(classifier, "feature_importances_"):
        return pd.Series()
    
    feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else np.arange(X_train.shape[1])
    importances = classifier.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    # Create and save plot
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    feat_imp.head(top_n).plot(kind='barh', title=f'Top {top_n} Feature Importances - {classifier_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = f'{classifier_name}_feature_importance_v{version}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return feat_imp

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
    version: float = 0.1
    decimal_places: int = 3
    
    def train_and_evaluate(self) -> Dict:
        """Train model and return evaluation metrics."""
        classifier_name = self.classifier.__class__.__name__.lower()
        logger.info(f'Training {classifier_name} model...')
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.classifier, self.X_train, self.y_train, 
                                  cv=skf, scoring='f1_weighted')
        
        # Train model
        self.classifier.fit(self.X_train, self.y_train)
        
        # Predictions
        train_pred = self.classifier.predict(self.X_train)
        test_pred = self.classifier.predict(self.X_test)
        train_proba = self.classifier.predict_proba(self.X_train)[:, 1]
        test_proba = self.classifier.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        metrics = self._calculate_metrics(train_pred, test_pred, train_proba, test_proba)
        metrics['cv_f1_weighted'] = np.round(cv_scores.mean(), self.decimal_places)
        
        # Feature importance
        feat_imp = save_feature_importance_plot(self.classifier, self.X_train, 
                                              classifier_name, self.version)
        
        # Classification report
        report_dict = classification_report(self.y_test, test_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Create metadata
        metadata = {
            'data_size': self.data_size,
            'classifier': classifier_name,
            'version': self.version,
            'params': extract_non_default_params(self.classifier),
            'feature_importance': feat_imp.to_dict() if not feat_imp.empty else {},
            **metrics
        }
        
        # Save artifacts
        model_path, meta_path, report_path = save_model_artifacts(
            self.classifier, metadata, classifier_name, self.version, report_df
        )
        
        # Log results
        self._log_results(classifier_name, metrics, test_pred)
        
        return {
            **metadata,
            'model_path': str(model_path),
            'metadata_path': str(meta_path),
            'report_path': str(report_path)
        }
    
    def _calculate_metrics(self, train_pred: np.ndarray, test_pred: np.ndarray, 
                         train_proba: np.ndarray, test_proba: np.ndarray) -> Dict:
        """Calculate all evaluation metrics."""
        return {
            'accuracy': {
                'train': np.round(accuracy_score(self.y_train, train_pred), self.decimal_places),
                'test': np.round(accuracy_score(self.y_test, test_pred), self.decimal_places)
            },
            'f1_weighted': {
                'train': np.round(f1_score(self.y_train, train_pred, average='weighted'), self.decimal_places),
                'test': np.round(f1_score(self.y_test, test_pred, average='weighted'), self.decimal_places)
            },
            'roc_auc': {
                'train': np.round(roc_auc_score(self.y_train, train_proba), self.decimal_places),
                'test': np.round(roc_auc_score(self.y_test, test_proba), self.decimal_places)
            }
        }
    
    def _log_results(self, classifier_name: str, metrics: Dict, test_pred: np.ndarray) -> None:
        """Log training results."""
        logger.info(f"\n{classifier_name.upper()} Results:")
        logger.info(f"CV F1 Weighted: {metrics['cv_f1_weighted']}")
        logger.info(f"Test Accuracy: {metrics['accuracy']['test']}")
        logger.info(f"Test F1 Weighted: {metrics['f1_weighted']['test']}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']['test']}")
        
        # Print detailed classification report
        print(f"\n{classifier_name.upper()} Classification Report:")
        print(classification_report(self.y_test, test_pred))

# Model Configuration
def get_model_configs(random_state: int = 42) -> Dict[str, ClassifierMixin]:
    """Get configured models for training."""
    return {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', random_state=random_state
        ),
        'xgboost': XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, 
            use_label_encoder=False, eval_metric='logloss', random_state=random_state
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=200, learning_rate=0.1, class_weight='balanced', 
            random_state=random_state, verbosity=-1
        )
    }

def train_multiple_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: np.ndarray, y_test: np.ndarray, 
                         data_size: int, version: float = 0.1) -> Dict:
    """Train multiple models and return results."""
    models = get_model_configs()
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        trainer = ModelTrainer(model, X_train, X_test, y_train, y_test, data_size, version)
        results[name] = trainer.train_and_evaluate()
    
    return results

def main() -> None:
    """Main ML pipeline execution."""
    try:
        # Setup data path
        data_path = Path("data/training/train.csv")
        
        # Validate data exists
        if not validate_data_path(data_path):
            return
        
        # Load and process data
        logger.info(f"Loading data from: {data_path}")
        data = load_data(data_path)
        
        logger.info(f"Data shape: {data.shape}")
        
        # Prepare train/test sets
        X_train, X_test, y_train, y_test = prepare_train_test_data(data)
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Train models
        results = train_multiple_models(X_train, X_test, y_train, y_test, len(data))
        
        logger.info("ML pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()