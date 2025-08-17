#!/usr/bin/env python3
"""
Pipeline Status Checker
Check what files exist and what steps have been completed
"""

import os
import pandas as pd

def check_pipeline_status():
    """Check the status of all pipeline steps"""
    
    print("🔍 CHECKING PIPELINE STATUS")
    print("="*50)
    
    # Check data files
    print("\n📊 DATA FILES:")
    data_files = [
        ('data/customers.csv', 'Customer data'),
        ('data/usage_data.csv', 'Usage data'), 
        ('data/support_tickets.csv', 'Support tickets'),
        ('data/processed_data.csv', 'Processed data'),
        ('data/featured_data.csv', 'Feature engineered data')
    ]
    
    for file_path, description in data_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ {description}: {len(df):,} rows")
            except:
                print(f"⚠️  {description}: File exists but can't read")
        else:
            print(f"❌ {description}: Missing")
    
    # Check model files
    print("\n🤖 MODEL FILES:")
    model_files = [
        ('models/random_forest_model.pkl', 'Trained model'),
        ('data/churn_risk_predictions.csv', 'Predictions'),
        ('data/feature_importance.csv', 'Feature importance'),
        ('plots/roc_curve.png', 'ROC curve plot'),
        ('plots/confusion_matrix.png', 'Confusion matrix plot')
    ]
    
    for file_path, description in model_files:
        if os.path.exists(file_path):
            print(f"✅ {description}: Available")
        else:
            print(f"❌ {description}: Missing")
    
    # Check notebook files
    print("\n📓 NOTEBOOK FILES:")
    notebook_files = [
        ('notebooks/01_data_exploration_and_cleaning.py', 'Data exploration script'),
        ('notebooks/02_feature_engineering_analysis.py', 'Feature engineering script'),
        ('notebooks/03_model_training_analysis.py', 'Model training script')
    ]
    
    for file_path, description in notebook_files:
        if os.path.exists(file_path):
            print(f"✅ {description}: Available")
        else:
            print(f"❌ {description}: Missing")
    
    # Pipeline status summary
    print("\n" + "="*50)
    print("📋 PIPELINE STATUS SUMMARY:")
    print("="*50)
    
    # Check if we have the minimum required files
    required_files = [
        'data/customers.csv',
        'data/featured_data.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if not missing_files:
        print("✅ READY FOR MODEL TRAINING!")
        print("   - All required data files are present")
        print("   - You can run: python src/models/train_model.py")
    else:
        print("❌ MISSING REQUIRED FILES:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 To fix this, run the pipeline in order:")
        print("   1. python src/data_pipeline/generate_sample_data.py")
        print("   2. python src/data_pipeline/main.py") 
        print("   3. python src/feature_engineering/feature_engineering.py")
        print("   4. python src/models/train_model.py")

if __name__ == "__main__":
    check_pipeline_status()
