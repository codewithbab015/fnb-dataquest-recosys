import json
import joblib
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from inspect import signature
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    f1_score, accuracy_score, 
    roc_auc_score, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# Function to save model metadata into the local 'models' folder
def save_model_and_metadata(model: ClassifierMixin, metadata: dict, classifier_name: str, version: float, report_df: pd.DataFrame):
    file_directory = Path('.').cwd().parent / "models/classifiers"
    file_directory.mkdir(exist_ok=True)

    model_path = file_directory / f"{classifier_name}_model_v{version}.pkl"
    joblib.dump(model, model_path)

    meta_path = file_directory / f"{classifier_name}_metadata_v{version}.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    report_path = file_directory / f"{classifier_name}_classifier_report_v{version}.csv"
    report_df.to_csv(report_path)

    logger.info(f"Saved model to: {model_path}")
    logger.info(f"Saved metadata to: {meta_path}")
    logger.info(f"Saved report classifier to: {report_path}")

    return model_path, meta_path, report_path

# Extract model parameters (non-defaults)
def model_non_default_params(model: BaseEstimator):
    sig = signature(model.__class__.__init__)
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not v.empty}
    current_params = model.get_params()
    return {k: v for k, v in current_params.items() if k in defaults and v != defaults[k]}

# Main model trainer
@dataclass
class ModelTrainer:
    classifier: ClassifierMixin | BaseEstimator
    X_train: np.ndarray | pd.DataFrame
    X_test: np.ndarray | pd.DataFrame
    y_train: np.ndarray | pd.Series
    y_test: np.ndarray | pd.Series
    data_size: int
    version: float = 0.1
    decimal: int = 3
    
    def train_model(self):
        classifier_name = self.classifier.__class__.__name__.lower()
        logger.info(f'Model training [{self.classifier.__class__.__name__.lower()}] started ... ')
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=skf, scoring='f1_weighted')

        print(f"CV Weighted F1 scores: {cv_scores}")
        print(f"Mean CV Weighted F1 score: {cv_scores.mean():.4f}")

        self.classifier.fit(self.X_train, self.y_train)
        train_y_pred = self.classifier.predict(self.X_train)
        train_y_pred_proba = self.classifier.predict_proba(self.X_train)[:, 1]
        test_y_pred = self.classifier.predict(self.X_test)
        test_y_pred_proba = self.classifier.predict_proba(self.X_test)[:, 1]

        train_acc_score = accuracy_score(self.y_train, train_y_pred)
        train_f_score = f1_score(self.y_train, train_y_pred, average='weighted')
        train_ra_score = roc_auc_score(self.y_train, train_y_pred_proba)

        test_acc_score = accuracy_score(self.y_test, test_y_pred)
        test_f_score = f1_score(self.y_test, test_y_pred, average='weighted')
        test_ra_score = roc_auc_score(self.y_test, test_y_pred_proba)

        if hasattr(self.classifier, "feature_importances_"):
            feature_names = self.X_train.columns if isinstance(self.X_train, pd.DataFrame) else np.arange(self.X_train.shape[1])
            importances = self.classifier.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            
            # Save to file
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 6))
            top_n = 20
            feat_imp.head(top_n).plot(kind='barh', title=f'Top {top_n} Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            file_img_path = f'{classifier_name}_top_feature_importances_v{self.version}.png'
            plt.savefig(file_img_path, dpi=300)
            plt.close()
        else:
            feat_imp = pd.Series()

        report_dict = classification_report(self.y_test, test_y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        metadata =  {
            'data_size': self.data_size, 
            'classifier': classifier_name,
            'params': model_non_default_params(self.classifier),
            'cv weighted f1': np.round(cv_scores.mean(), self.decimal),
            'accuracy score': {'train': np.round(train_acc_score, self.decimal), 'test': np.round(test_acc_score, self.decimal)},
            'weighted f1 score': {'train': np.round(train_f_score, self.decimal), 'test': np.round(test_f_score, self.decimal)},
            'roc-auc score': {'train': np.round(train_ra_score, self.decimal), 'test': np.round(test_ra_score, self.decimal)},
            'feature importance': feat_imp.to_dict() if not feat_imp.empty else {},
        }
        model_path, meta_path, report_path = save_model_and_metadata(self.classifier, metadata,classifier_name, self.version, report_df)

        print(f"\nResults for {classifier_name}:")
        print("Accuracy Score:", train_acc_score)
        print("F1 Score:", train_f_score)
        print("Roc-Auc Score:", train_ra_score)
        print(classification_report(self.y_test, test_y_pred))

        return {
           **metadata,
            'model path': model_path,
            'metdata path': meta_path,
            'report path': report_path
        }



def run_models(*args) -> None:
    # Compare model performance metrics using Logistic Regression as the baseline
    seed = 43
    models = {
        'Logistic Regression': LogisticRegression(max_iter=100, random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=seed),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.1, class_weight='balanced', random_state=seed, verbosity=-1)
    }
    
    X_train, X_test, y_train, y_test = args

    # Train, predict, and evaluate
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model_trainer = ModelTrainer(model, X_train, X_test, y_train, y_test, version=0.1)
        trainer = model_trainer.train_model()