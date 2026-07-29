import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_raw_data(data_path):
    """
    Loads raw CSV data and drops duplicate rows.
    """
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    return df

def split_data(df, target_col='Spoilage_Risk', test_size=0.2, random_state=42):
    """
    Splits the dataframe into X (features) and y (continuous target) before any 
    preprocessing or categorization to avoid data leakage.
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train_cont, y_test_cont = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train_cont, y_test_cont

def create_target_labels(y_train_cont, y_test_cont):
    """
    Creates target labels (Low Risk, Medium Risk, High Risk) using ONLY the training set
    percentiles (33rd and 66th percentiles) to avoid target leakage.
    """
    t33 = y_train_cont.quantile(0.33)
    t66 = y_train_cont.quantile(0.66)
    
    def apply_thresholds(y_val):
        conditions = [
            (y_val <= t33),
            (y_val > t33) & (y_val <= t66),
            (y_val > t66)
        ]
        choices = ['Low Risk', 'Medium Risk', 'High Risk']
        return np.select(conditions, choices, default='Medium Risk')

    y_train_cat = pd.Series(apply_thresholds(y_train_cont), index=y_train_cont.index, name='Risk_Category')
    y_test_cat = pd.Series(apply_thresholds(y_test_cont), index=y_test_cont.index, name='Risk_Category')
    
    thresholds = {
        '33rd_percentile': float(t33),
        '66th_percentile': float(t66)
    }
    
    return y_train_cat, y_test_cat, thresholds
