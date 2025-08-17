#!/usr/bin/env python3
"""
Churn Prediction Pipeline - Main Execution Script
Runs the complete pipeline from data generation to dashboard launch
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"Error: {e.stderr}")
        return False

def check_file_exists(file_path):
    """Check if a file exists"""
    return os.path.exists(file_path)

def main():
    """Main pipeline execution"""
    
    print_header("Customer Churn Prediction & Retention System")
    print("Complete pipeline execution from data generation to dashboard")
    
    # Step 1: Install dependencies
    print_step(1, "Installing Dependencies")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install dependencies. Please check requirements.txt")
        return False
    
    # Step 2: Generate sample data
    print_step(2, "Generating Sample Data")
    if not run_command("python src/data_pipeline/generate_sample_data.py", "Generating customer data"):
        print("❌ Failed to generate sample data")
        return False
    
    # Step 3: Run data pipeline
    print_step(3, "Running Data Pipeline")
    if not run_command("python src/data_pipeline/main.py", "Processing data pipeline"):
        print("❌ Failed to run data pipeline")
        return False
    
    # Step 4: Feature engineering
    print_step(4, "Feature Engineering")
    if not run_command("python src/feature_engineering/feature_engineering.py", "Feature engineering"):
        print("❌ Failed to complete feature engineering")
        return False
    
    # Step 5: Train model
    print_step(5, "Training Machine Learning Model")
    if not run_command("python src/models/train_model.py", "Model training"):
        print("❌ Failed to train model")
        return False
    
    # Step 6: Verify outputs
    print_step(6, "Verifying Pipeline Outputs")
    
    required_files = [
        "data/customers.csv",
        "data/usage_data.csv", 
        "data/support_tickets.csv",
        "data/processed_data.csv",
        "data/featured_data.csv",
        "data/churn_risk_predictions.csv",
        "models/random_forest_model.pkl",
        "models/feature_importance.csv"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if check_file_exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            all_files_exist = False
    
    if not all_files_exist:
        print("❌ Some required files are missing. Pipeline may have failed.")
        return False
    
    # Step 7: Launch dashboard
    print_step(7, "Launching Interactive Dashboard")
    print("🎯 Dashboard will open in your browser at http://localhost:8501")
    print("📊 You can now explore the churn prediction results!")
    print("\nTo stop the dashboard, press Ctrl+C in the terminal")
    
    # Launch dashboard
    try:
        subprocess.run("streamlit run src/dashboard/app.py", shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    
    return True

def quick_demo():
    """Quick demo mode - skip data generation if files exist"""
    print_header("Quick Demo Mode")
    print("Running pipeline with existing data (if available)")
    
    # Check if data already exists
    if check_file_exists("data/churn_risk_predictions.csv") and check_file_exists("models/random_forest_model.pkl"):
        print("✅ Found existing model and predictions")
        print("🚀 Launching dashboard directly...")
        
        try:
            subprocess.run("streamlit run src/dashboard/app.py", shell=True)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        return True
    else:
        print("❌ No existing data found. Running full pipeline...")
        return main()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        quick_demo()
    else:
        print("Usage:")
        print("  python run_pipeline.py          # Run full pipeline")
        print("  python run_pipeline.py --demo   # Quick demo (if data exists)")
        print("\nStarting full pipeline...")
        main()
