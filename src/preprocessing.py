import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FoodLossPreprocessor(BaseEstimator, TransformerMixin):
    """
    Stateful preprocessor for the Post-Harvest Food Loss Risk Prediction dataset.
    Recovers clean scales from exponentiated features, imputes, encodes, and scales.
    """
    def __init__(self):
        # Columns to drop due to extreme missingness (100% or near 100% infinite)
        self.cols_to_drop = [
            'Inventory_Levels', 
            'Vehicle_Load_Capacity', 
            'Crop_Yield', 
            'Station_Capacity', 
            'Operational_Cost', 
            'Energy_Consumption', 
            'Efficiency_Ratio'
        ]
        # Columns to drop because they are metadata/IDs and not predictive features
        self.meta_cols_to_drop = ['Unnamed: 0', 'Harvest_Date']
        
        # We will learn these during fit
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.modes = {}
        self.one_hot_categories = {}
        self.means = None
        self.stds = None
        self.feature_names_out = None

    def fit(self, X, y=None):
        X_clean = X.copy()
        
        # Drop columns
        all_drops = [col for col in (self.cols_to_drop + self.meta_cols_to_drop) if col in X_clean.columns]
        X_clean = X_clean.drop(columns=all_drops)
        
        # Identify column types
        self.numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Replace inf with nan for median calculation
        X_temp = X_clean.copy()
        X_temp[self.numeric_cols] = X_temp[self.numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Apply log1p before calculating median to ensure stable medians in log-scale
        for col in self.numeric_cols:
            X_temp[col] = np.log1p(X_temp[col])
            
        # Store medians for numeric columns
        for col in self.numeric_cols:
            # Fallback to 0 if median is nan (should not happen)
            self.medians[col] = float(X_temp[col].median()) if not np.isnan(X_temp[col].median()) else 0.0
            
        # Store modes for categorical columns
        for col in self.categorical_cols:
            if not X_temp[col].dropna().empty:
                self.modes[col] = X_temp[col].mode().iloc[0]
            else:
                self.modes[col] = "Unknown"
                
            # Store unique categories for OHE
            self.one_hot_categories[col] = X_temp[col].dropna().unique().tolist()
            
        # Transform temporary df to calculate mean and std for standard scaling
        X_imputed = X_temp.copy()
        for col in self.numeric_cols:
            X_imputed[col] = X_imputed[col].fillna(self.medians[col])
            
        for col in self.categorical_cols:
            X_imputed[col] = X_imputed[col].fillna(self.modes[col])
            
        # Calculate mean and std for standardization
        self.means = X_imputed[self.numeric_cols].mean().to_dict()
        self.stds = X_imputed[self.numeric_cols].std().to_dict()
        for col in self.numeric_cols:
            if self.stds[col] == 0.0 or np.isnan(self.stds[col]):
                self.stds[col] = 1.0 # Avoid division by zero
                
        # Determine output feature names
        feature_names = list(self.numeric_cols)
        for col in self.categorical_cols:
            for cat in sorted(self.one_hot_categories[col]):
                feature_names.append(f"{col}_{cat}")
        self.feature_names_out = feature_names
        
        return self

    def transform(self, X):
        X_clean = X.copy()
        
        # Drop columns
        all_drops = [col for col in (self.cols_to_drop + self.meta_cols_to_drop) if col in X_clean.columns]
        X_clean = X_clean.drop(columns=all_drops)
        
        # Replace inf with nan
        X_clean[self.numeric_cols] = X_clean[self.numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Apply log reconstruction
        for col in self.numeric_cols:
            X_clean[col] = np.log1p(X_clean[col])
            
        # Impute numeric columns
        for col in self.numeric_cols:
            X_clean[col] = X_clean[col].fillna(self.medians[col])
            
        # Impute categorical columns
        for col in self.categorical_cols:
            X_clean[col] = X_clean[col].fillna(self.modes[col])
            
        # Standardize numeric columns
        for col in self.numeric_cols:
            X_clean[col] = (X_clean[col] - self.means[col]) / self.stds[col]
            
        # One-hot encode categorical columns
        encoded_dfs = []
        for col in self.categorical_cols:
            categories = sorted(self.one_hot_categories[col])
            for cat in categories:
                col_name = f"{col}_{cat}"
                X_clean[col_name] = (X_clean[col] == cat).astype(float)
                
        # Drop original categorical columns
        X_clean = X_clean.drop(columns=self.categorical_cols)
        
        # Ensure we have all columns in the correct order, filled with 0 if missing
        for col_out in self.feature_names_out:
            if col_out not in X_clean.columns:
                X_clean[col_out] = 0.0
                
        return X_clean[self.feature_names_out].copy()

    def get_feature_names_out(self):
        return self.feature_names_out
