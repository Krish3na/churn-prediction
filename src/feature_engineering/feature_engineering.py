"""
Feature Engineering for Churn Prediction
Prepares and transforms features for machine learning model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib
import os

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        
    def load_data(self, data_path='data/processed_data.csv'):
        """Load processed data"""
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        print(f"Data shape: {data.shape}")
        return data
    
    def handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Fill numeric columns with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        print(f"Missing values handled. Remaining nulls: {data.isnull().sum().sum()}")
        return data
    
    def create_derived_features(self, data):
        """Create additional derived features"""
        print("Creating derived features...")
        
        # Customer engagement score
        data['engagement_score'] = (
            data['usage_intensity_30d'] * 0.4 +
            data['usage_intensity_60d'] * 0.3 +
            data['usage_intensity_90d'] * 0.3
        )
        
        # Support satisfaction score
        data['support_satisfaction_score'] = (
            data['satisfaction_mean_30d'] * 0.5 +
            data['satisfaction_mean_60d'] * 0.3 +
            data['satisfaction_mean_90d'] * 0.2
        )
        
        # Risk indicators
        data['high_risk_indicator'] = (
            (data['support_intensity_30d'] > 0.1).astype(int) +
            (data['satisfaction_mean_30d'] < 6).astype(int) +
            (data['usage_intensity_30d'] < 0.3).astype(int) +
            (data['late_payments'] > 1).astype(int)
        )
        
        # Revenue efficiency
        data['revenue_per_feature'] = data['monthly_revenue'] / (data['features_used'] + 1)
        data['revenue_per_login'] = data['monthly_revenue'] / (data['monthly_logins'] + 1)
        
        # Usage consistency (lower std means more consistent usage)
        data['usage_consistency_30d'] = 1 / (data['logins_std_30d'] + 1)
        data['usage_consistency_60d'] = 1 / (data['logins_std_60d'] + 1)
        data['usage_consistency_90d'] = 1 / (data['logins_std_90d'] + 1)
        
        # Support efficiency
        data['support_efficiency'] = data['resolution_time_mean_30d'] / (data['tickets_count_30d'] + 1)
        
        # Plan value perception
        plan_values = {'Starter': 1, 'Basic': 2, 'Premium': 3, 'Enterprise': 4}
        data['plan_value'] = data['plan'].map(plan_values)
        data['value_perception'] = data['features_used'] / data['plan_value']
        
        print(f"Created {len([col for col in data.columns if col not in ['customer_id', 'join_date', 'churn_date', 'churned']])} features")
        return data
    
    def encode_categorical_features(self, data):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_columns = ['plan', 'country', 'industry', 'payment_method']
        
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        return data
    
    def select_features(self, data, target_col='churned', n_features=50):
        """Select the most important features"""
        print(f"Selecting top {n_features} features...")
        
        # Prepare feature matrix (exclude non-feature columns)
        exclude_cols = ['customer_id', 'join_date', 'churn_date', 'churned', 'churn_score']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].select_dtypes(include=[np.number])
        y = data[target_col]
        
        # Remove columns with zero variance
        X = X.loc[:, X.var() > 0]
        
        # Feature selection using ANOVA F-test
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        print(f"Selected {len(self.selected_features)} features")
        return self.selected_features
    
    def scale_features(self, data, selected_features):
        """Scale numerical features"""
        print("Scaling features...")
        
        # Scale selected features
        data_scaled = data.copy()
        data_scaled[selected_features] = self.scaler.fit_transform(data[selected_features])
        
        return data_scaled
    
    def prepare_final_dataset(self, data, selected_features):
        """Prepare final dataset for modeling"""
        print("Preparing final dataset...")
        
        # Select only the features we need
        final_features = ['customer_id'] + selected_features + ['churned']
        final_data = data[final_features].copy()
        
        # Ensure no missing values
        final_data = final_data.dropna()
        
        print(f"Final dataset shape: {final_data.shape}")
        return final_data
    
    def save_encoders(self, output_dir='models'):
        """Save encoders and scalers for later use"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save label encoders
        for col, encoder in self.label_encoders.items():
            joblib.dump(encoder, f'{output_dir}/label_encoder_{col}.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        
        # Save feature selector
        if self.feature_selector:
            joblib.dump(self.feature_selector, f'{output_dir}/feature_selector.pkl')
        
        # Save selected features list
        if self.selected_features:
            joblib.dump(self.selected_features, f'{output_dir}/selected_features.pkl')
        
        print(f"Encoders and scalers saved to {output_dir}")
    
    def get_feature_importance_scores(self, data, selected_features):
        """Get feature importance scores for analysis"""
        if self.feature_selector:
            feature_scores = pd.DataFrame({
                'feature': selected_features,
                'score': self.feature_selector.scores_[:len(selected_features)],
                'p_value': self.feature_selector.pvalues_[:len(selected_features)]
            })
            return feature_scores.sort_values('score', ascending=False)
        return None
    
    def process_data(self, data_path='data/processed_data.csv', output_path='data/featured_data.csv'):
        """Complete feature engineering pipeline"""
        
        # Load data
        data = self.load_data(data_path)
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Create derived features
        data = self.create_derived_features(data)
        
        # Encode categorical features
        data = self.encode_categorical_features(data)
        
        # Select features
        selected_features = self.select_features(data)
        
        # Scale features
        data_scaled = self.scale_features(data, selected_features)
        
        # Prepare final dataset
        final_data = self.prepare_final_dataset(data_scaled, selected_features)
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_data.to_csv(output_path, index=False)
        print(f"Featured data saved to: {output_path}")
        
        # Save encoders
        self.save_encoders()
        
        # Get feature importance
        feature_importance = self.get_feature_importance_scores(data, selected_features)
        if feature_importance is not None:
            feature_importance.to_csv('data/feature_importance.csv', index=False)
            print("Top 10 most important features:")
            print(feature_importance.head(10))
        
        return final_data, selected_features

def main():
    """Main feature engineering execution"""
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Process data
    final_data, selected_features = fe.process_data()
    
    print("\n=== Feature Engineering Complete ===")
    print(f"Final dataset shape: {final_data.shape}")
    print(f"Number of features: {len(selected_features)}")
    print(f"Churn rate: {final_data['churned'].mean():.2%}")

if __name__ == "__main__":
    main()
