import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure the root directory is in the python path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

from src.data_loader import load_raw_data, split_data, create_target_labels
from src.preprocessing import FoodLossPreprocessor

# Add proper exception handling if XGBoost or CatBoost is unavailable
try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False

try:
    from catboost import CatBoostClassifier
    has_catboost = True
except ImportError:
    has_catboost = False


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to ensure targets are handled as strings, and that models without 
    decision_function (like RandomForest, XGBoost, CatBoost) are fully compatible 
    with app.py (exposing classes_, decision_function, and feature importances/coefficients).
    """
    def __init__(self, estimator, label_encoder):
        self.estimator = estimator
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def fit(self, X, y, **fit_params):
        y_encoded = self.label_encoder.transform(y)
        self.estimator.fit(X, y_encoded, **fit_params)
        return self

    def predict(self, X):
        preds_encoded = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(preds_encoded)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.estimator, 'decision_function'):
            return self.estimator.decision_function(X)
        else:
            probs = self.estimator.predict_proba(X)
            return np.log(probs + 1e-15)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    @property
    def coef_(self):
        return self.estimator.coef_


def train_pipeline(data_path="data/EuroCrop_agricultural_logistics_dataset.csv", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("Loading data...")
    df = load_raw_data(data_path)
    
    print("Splitting features and continuous target...")
    X_train, X_test, y_train_cont, y_test_cont = split_data(df)
    
    print("Creating target labels using training percentiles...")
    y_train_cat, y_test_cat, thresholds = create_target_labels(y_train_cont, y_test_cont)
    
    print("Fitting preprocessor...")
    preprocessor = FoodLossPreprocessor()
    preprocessor.fit(X_train)
    
    X_train_clean = preprocessor.transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    
    # Target encoding to ensure integer targets for all classifiers (especially XGBoost)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_cat)
    y_test_encoded = le.transform(y_test_cat)
    
    # Models to train and compare
    models = {
        'DummyClassifier': DummyClassifier(strategy='stratified', random_state=42),
        'RidgeClassifier': RidgeClassifier(random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    }
    
    # Exception handling for XGBoost
    if has_xgboost:
        models['XGBClassifier'] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss"
        )
    else:
        print("Warning: XGBoost is unavailable and will not be compared.")
        
    # Exception handling for CatBoost
    if has_catboost:
        models['CatBoostClassifier'] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False
        )
    else:
        print("Warning: CatBoost is unavailable and will not be compared.")
    
    model_comparison = []
    trained_models = {}
    model_f1s = {}
    
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train_clean, y_train_encoded)
        preds = model.predict(X_test_clean)
        
        acc = accuracy_score(y_test_encoded, preds)
        f1 = f1_score(y_test_encoded, preds, average='macro')
        print(f"  {name:30} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        
        model_comparison.append({
            'Model': name,
            'Accuracy': acc,
            'Macro_F1': f1
        })
        
        trained_models[name] = model
        model_f1s[name] = f1
        
    # Determine the best boosting model to tune between GradientBoosting and XGBoost
    boosting_candidates = ['GradientBoostingClassifier']
    if has_xgboost:
        boosting_candidates.append('XGBClassifier')
        
    best_boosting_name = max(boosting_candidates, key=lambda x: model_f1s[x])
    print(f"\nBest boosting model selected for hyperparameter tuning: {best_boosting_name}")
    
    # Define hyperparameter grid for RandomizedSearchCV
    if best_boosting_name == 'GradientBoostingClassifier':
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_boosting_model = GradientBoostingClassifier(random_state=42)
    else:  # XGBClassifier
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        base_boosting_model = XGBClassifier(
            random_state=42,
            eval_metric="mlogloss"
        )
        
    print(f"Running RandomizedSearchCV on {best_boosting_name} (5-fold CV)...")
    random_search = RandomizedSearchCV(
        estimator=base_boosting_model,
        param_distributions=param_dist,
        n_iter=15,
        scoring='f1_macro',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train_clean, y_train_encoded)
    tuned_model = random_search.best_estimator_
    tuned_name = f"Tuned_{best_boosting_name}"
    
    # Evaluate the tuned model
    tuned_preds = tuned_model.predict(X_test_clean)
    tuned_acc = accuracy_score(y_test_encoded, tuned_preds)
    tuned_f1 = f1_score(y_test_encoded, tuned_preds, average='macro')
    print(f"  {tuned_name:30} | Accuracy: {tuned_acc:.4f} | Macro F1: {tuned_f1:.4f}")
    
    model_comparison.append({
        'Model': tuned_name,
        'Accuracy': tuned_acc,
        'Macro_F1': tuned_f1
    })
    
    trained_models[tuned_name] = tuned_model
    model_f1s[tuned_name] = tuned_f1

    # Select the model with the highest Macro F1, excluding DummyClassifier
    best_model_name = None
    best_f1 = -1.0
    
    for name, f1 in model_f1s.items():
        if name != 'DummyClassifier' and f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    best_raw_model = trained_models[best_model_name]
    print(f"\nOverall best model selected: {best_model_name} (Macro F1: {best_f1:.4f})")
    
    # Save comparison report
    comparison_df = pd.DataFrame(model_comparison)
    comparison_df.to_csv("reports/model_comparison.csv", index=False)
    print("Saved reports/model_comparison.csv")
    
    # Wrap the best model for target decoding and decision_function compatibility
    wrapped_best_model = ClassifierWrapper(best_raw_model, le)

    # Evaluate best wrapped model on test set and save metrics.json
    best_preds_cat = wrapped_best_model.predict(X_test_clean)
    
    accuracy = float(accuracy_score(y_test_cat, best_preds_cat))
    macro_precision = float(precision_score(y_test_cat, best_preds_cat, average='macro', zero_division=0))
    macro_recall = float(recall_score(y_test_cat, best_preds_cat, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_test_cat, best_preds_cat, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_test_cat, best_preds_cat, average='weighted', zero_division=0))
    high_risk_recall = float(recall_score(y_test_cat, best_preds_cat, labels=['High Risk'], average=None, zero_division=0)[0])

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "high_risk_recall": high_risk_recall
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved reports/metrics.json")
    
    # Compute raw data statistics for recommendations and Streamlit widgets
    raw_stats = {}
    for col in X_train.select_dtypes(include=[np.number]).columns:
        valid_series = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not valid_series.empty:
            raw_stats[col] = {
                'min': float(valid_series.min()),
                'max': float(valid_series.max()),
                'mean': float(valid_series.mean()),
                '25th': float(valid_series.quantile(0.25)),
                '50th': float(valid_series.quantile(0.50)),
                '75th': float(valid_series.quantile(0.75))
            }
            
    # Save metadata.joblib
    metadata = {
        'thresholds': thresholds,
        'medians': preprocessor.medians,
        'modes': preprocessor.modes,
        'numeric_cols': preprocessor.numeric_cols,
        'categorical_cols': preprocessor.categorical_cols,
        'one_hot_categories': preprocessor.one_hot_categories,
        'feature_names_out': preprocessor.feature_names_out,
        'raw_stats': raw_stats,
        'best_model_name': best_model_name
    }
    joblib.dump(metadata, os.path.join(models_dir, "metadata.joblib"))
    print("Saved models/metadata.joblib")
    
    # Save the full Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', wrapped_best_model)
    ])
    joblib.dump(full_pipeline, os.path.join(models_dir, "food_loss_pipeline.joblib"))
    print("Saved models/food_loss_pipeline.joblib")
    
    # Save individual best model for compatibility
    joblib.dump(wrapped_best_model, os.path.join(models_dir, "best_model.joblib"))
    print("Saved models/best_model.joblib")

if __name__ == '__main__':
    train_pipeline()
