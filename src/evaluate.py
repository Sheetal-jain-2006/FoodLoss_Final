import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the root directory is in the python path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data_loader import load_raw_data, split_data

def evaluate_pipeline(data_path="data/EuroCrop_agricultural_logistics_dataset.csv", models_dir="models", reports_dir="reports"):
    os.makedirs(reports_dir, exist_ok=True)
    
    print("Loading test data...")
    df = load_raw_data(data_path)
    _, X_test, _, y_test_cont = split_data(df)
    
    print("Loading metadata to retrieve target thresholds...")
    metadata_path = os.path.join(models_dir, "metadata.joblib")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run train.py first.")
        
    metadata = joblib.load(metadata_path)
    t33 = metadata['thresholds']['33rd_percentile']
    t66 = metadata['thresholds']['66th_percentile']
    
    # Categorize target using training thresholds to prevent leakage
    conditions = [
        (y_test_cont <= t33),
        (y_test_cont > t33) & (y_test_cont <= t66),
        (y_test_cont > t66)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk']
    y_test_cat = pd.Series(np.select(conditions, choices, default='Medium Risk'), index=y_test_cont.index)
    
    # Load pipeline
    pipeline_path = os.path.join(models_dir, "food_loss_pipeline.joblib")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}. Please run train.py first.")
        
    pipeline = joblib.load(pipeline_path)
    
    print("Making predictions using serialized pipeline...")
    preds = pipeline.predict(X_test)
    
    # Compute metrics
    print("Calculating evaluation metrics...")
    acc = accuracy_score(y_test_cat, preds)
    macro_prec = precision_score(y_test_cat, preds, average='macro', zero_division=0)
    macro_rec = recall_score(y_test_cat, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test_cat, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test_cat, preds, average='weighted', zero_division=0)
    
    clf_rep = classification_report(y_test_cat, preds, output_dict=True, zero_division=0)
    high_risk_recall = clf_rep.get('High Risk', {}).get('recall', 0.0)
    
    metrics = {
        'accuracy': float(acc),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'high_risk_recall': float(high_risk_recall)
    }
    
    # Save metrics.json
    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved reports/metrics.json")
    
    print("\nClassification Report:")
    print(classification_report(y_test_cat, preds, zero_division=0))
    
    # Confusion Matrix
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_test_cat, preds, labels=['Low Risk', 'Medium Risk', 'High Risk'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Risk', 'Medium Risk', 'High Risk'], 
                yticklabels=['Low Risk', 'Medium Risk', 'High Risk'])
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved reports/confusion_matrix.png")
    
    # Extract Feature Importances
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    feature_names = preprocessor.get_feature_names_out()
    
    importances = None
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.mean(np.abs(classifier.coef_), axis=0)
        
    if importances is not None:
        print("Extracting feature importances...")
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        feat_imp_df = pd.DataFrame({'Feature': feat_imp.index, 'Importance': feat_imp.values})
        feat_imp_df.to_csv(os.path.join(reports_dir, "feature_importance.csv"), index=False)
        print("Saved reports/feature_importance.csv")
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "feature_importance.png"), dpi=150)
        plt.close()
        print("Saved reports/feature_importance.png")
    else:
        print("The best classifier does not support feature importances or coefficient extraction.")

if __name__ == '__main__':
    evaluate_pipeline()
