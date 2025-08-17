"""
Main Data Pipeline for Churn Prediction
Extracts data from database and prepares for feature engineering
"""

import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.generate_sample_data import create_database

class DataPipeline:
    def __init__(self, db_path='data/churn_prediction.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect_database(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
        return True
    
    def extract_customer_data(self):
        """Extract customer data from database"""
        query = """
        SELECT * FROM customers
        """
        return pd.read_sql_query(query, self.conn)
    
    def extract_usage_data(self):
        """Extract usage data from database"""
        query = """
        SELECT * FROM usage_data
        """
        return pd.read_sql_query(query, self.conn)
    
    def extract_support_tickets(self):
        """Extract support ticket data from database"""
        query = """
        SELECT * FROM support_tickets
        """
        return pd.read_sql_query(query, self.conn)
    
    def calculate_usage_features(self, usage_df):
        """Calculate aggregated usage features per customer"""
        
        # Convert date columns
        usage_df['usage_date'] = pd.to_datetime(usage_df['usage_date'])
        
        # Calculate usage features for last 30, 60, and 90 days
        current_date = datetime.now()
        
        usage_features = []
        
        for period in [30, 60, 90]:
            period_start = current_date - timedelta(days=period)
            period_data = usage_df[usage_df['usage_date'] >= period_start]
            
            # Group by customer and calculate features
            period_features = period_data.groupby('customer_id').agg({
                'daily_logins': ['sum', 'mean', 'std'],
                'session_duration_minutes': ['sum', 'mean', 'std'],
                'pages_viewed': ['sum', 'mean', 'std'],
                'features_accessed': ['sum', 'mean', 'std'],
                'usage_date': 'count'  # Number of usage days
            }).reset_index()
            
            # Flatten column names
            period_features.columns = [
                'customer_id',
                f'logins_sum_{period}d', f'logins_mean_{period}d', f'logins_std_{period}d',
                f'session_duration_sum_{period}d', f'session_duration_mean_{period}d', f'session_duration_std_{period}d',
                f'pages_viewed_sum_{period}d', f'pages_viewed_mean_{period}d', f'pages_viewed_std_{period}d',
                f'features_accessed_sum_{period}d', f'features_accessed_mean_{period}d', f'features_accessed_std_{period}d',
                f'usage_days_{period}d'
            ]
            
            usage_features.append(period_features)
        
        # Merge all period features
        final_features = usage_features[0]
        for features in usage_features[1:]:
            final_features = final_features.merge(features, on='customer_id', how='left')
        
        return final_features
    
    def calculate_support_features(self, tickets_df):
        """Calculate aggregated support ticket features per customer"""
        
        # Convert date columns
        tickets_df['ticket_date'] = pd.to_datetime(tickets_df['ticket_date'])
        
        # Calculate support features for last 30, 60, and 90 days
        current_date = datetime.now()
        
        support_features = []
        
        for period in [30, 60, 90]:
            period_start = current_date - timedelta(days=period)
            period_data = tickets_df[tickets_df['ticket_date'] >= period_start]
            
            # Group by customer and calculate features
            period_features = period_data.groupby('customer_id').agg({
                'ticket_id': 'count',  # Number of tickets
                'resolution_time_hours': ['mean', 'std'],
                'customer_satisfaction': ['mean', 'std'],
                'priority': lambda x: (x == 'High').sum() + (x == 'Critical').sum() * 2,  # High priority tickets
                'status': lambda x: (x == 'Open').sum() + (x == 'In Progress').sum()  # Open tickets
            }).reset_index()
            
            # Flatten column names
            period_features.columns = [
                'customer_id',
                f'tickets_count_{period}d',
                f'resolution_time_mean_{period}d', f'resolution_time_std_{period}d',
                f'satisfaction_mean_{period}d', f'satisfaction_std_{period}d',
                f'high_priority_tickets_{period}d',
                f'open_tickets_{period}d'
            ]
            
            support_features.append(period_features)
        
        # Merge all period features
        final_features = support_features[0]
        for features in support_features[1:]:
            final_features = final_features.merge(features, on='customer_id', how='left')
        
        return final_features
    
    def create_feature_dataset(self):
        """Create the complete feature dataset"""
        
        print("Extracting customer data...")
        customers_df = self.extract_customer_data()
        
        print("Extracting usage data...")
        usage_df = self.extract_usage_data()
        
        print("Extracting support ticket data...")
        tickets_df = self.extract_support_tickets()
        
        print("Calculating usage features...")
        usage_features = self.calculate_usage_features(usage_df)
        
        print("Calculating support features...")
        support_features = self.calculate_support_features(tickets_df)
        
        print("Merging all features...")
        # Merge all datasets
        final_dataset = customers_df.merge(usage_features, on='customer_id', how='left')
        final_dataset = final_dataset.merge(support_features, on='customer_id', how='left')
        
        # Fill NaN values
        numeric_columns = final_dataset.select_dtypes(include=[np.number]).columns
        final_dataset[numeric_columns] = final_dataset[numeric_columns].fillna(0)
        
        # Add derived features
        final_dataset['tenure_days'] = (datetime.now() - pd.to_datetime(final_dataset['join_date'])).dt.days
        final_dataset['revenue_per_day'] = final_dataset['monthly_revenue'] / 30
        
        # Calculate usage intensity
        final_dataset['usage_intensity_30d'] = final_dataset['usage_days_30d'] / 30
        final_dataset['usage_intensity_60d'] = final_dataset['usage_days_60d'] / 60
        final_dataset['usage_intensity_90d'] = final_dataset['usage_days_90d'] / 90
        
        # Calculate support intensity
        final_dataset['support_intensity_30d'] = final_dataset['tickets_count_30d'] / 30
        final_dataset['support_intensity_60d'] = final_dataset['tickets_count_60d'] / 60
        final_dataset['support_intensity_90d'] = final_dataset['tickets_count_90d'] / 90
        
        print(f"Final dataset shape: {final_dataset.shape}")
        print(f"Churn rate: {final_dataset['churned'].mean():.2%}")
        
        return final_dataset
    
    def save_processed_data(self, dataset, output_path='data/processed_data.csv'):
        """Save processed dataset to CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

def main():
    """Main pipeline execution"""
    
    # Check if database exists, if not create it
    if not os.path.exists('data/churn_prediction.db'):
        print("Database not found. Creating sample database...")
        create_database()
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Connect to database
    if not pipeline.connect_database():
        return
    
    try:
        # Create feature dataset
        dataset = pipeline.create_feature_dataset()
        
        # Save processed data
        pipeline.save_processed_data(dataset)
        
        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Total customers: {len(dataset)}")
        print(f"Churned customers: {dataset['churned'].sum()}")
        print(f"Churn rate: {dataset['churned'].mean():.2%}")
        print(f"Features: {len(dataset.columns)}")
        
        print("\n=== Feature Overview ===")
        print("Customer features:", [col for col in dataset.columns if col.startswith(('customer_id', 'join_date', 'plan', 'country', 'industry', 'monthly_revenue', 'churned'))])
        print("Usage features:", [col for col in dataset.columns if 'logins' in col or 'session' in col or 'pages' in col or 'features_accessed' in col or 'usage_days' in col])
        print("Support features:", [col for col in dataset.columns if 'tickets' in col or 'resolution' in col or 'satisfaction' in col or 'priority' in col])
        
    finally:
        pipeline.close_connection()

if __name__ == "__main__":
    main()
