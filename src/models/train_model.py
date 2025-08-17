"""
Random Forest Model Training for Churn Prediction
Trains and evaluates Random Forest model with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.best_params = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def load_data(self, data_path='data/featured_data.csv'):
        """Load featured data"""
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        print(f"Data shape: {data.shape}")
        return data
    
    def prepare_data(self, data):
        """Prepare data for training"""
        print("Preparing data for training...")
        
        # Separate features and target
        X = data.drop(['customer_id', 'churned'], axis=1)
        y = data['churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training churn rate: {y_train.mean():.2%}")
        print(f"Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, tune_hyperparameters=True):
        """Train Random Forest model with optional hyperparameter tuning"""
        
        if tune_hyperparameters:
            print("\n" + "🔧"*20)
            print("🔧 STARTING HYPERPARAMETER TUNING")
            print("🔧"*20)
            print("This will test different parameter combinations to find the best model...")
            print("Estimated time: 2-5 minutes depending on your computer")
            print("You can press Ctrl+C to stop and use default parameters")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample']
                # 'n_estimators': [100, 200],
                # 'max_depth': [15, 25, None],
                # 'min_samples_split': [2, 5],
                # 'min_samples_leaf': [1, 2],
                # 'max_features': ['sqrt', 'log2'],
                # 'class_weight': ['balanced']
            }
            
            # Calculate total combinations
            total_combinations = 1
            for param, values in param_grid.items():
                total_combinations *= len(values)
            print(f"Testing {total_combinations} parameter combinations with 5-fold cross-validation...")
            print(f"Total model fits: {total_combinations * 5}")
            
            # Initialize base model
            base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=base_rf,
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            print("✅ Hyperparameter tuning completed successfully!")
            
        else:
            print("Training Random Forest with default parameters...")
            
            # Train with default parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def get_feature_importance(self, feature_names):
        """Get and display feature importance"""
        print("Analyzing feature importance...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature importance DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("Top 15 most important features:")
        print(self.feature_importance.head(15))
        
        return self.feature_importance
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("No feature importance data available")
            return
        
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, output_dir='models'):
        """Save trained model and related files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f'{output_dir}/random_forest_model.pkl')
        
        # Save feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        
        # Save best parameters
        if self.best_params is not None:
            joblib.dump(self.best_params, f'{output_dir}/best_params.pkl')
        
        print(f"Model and related files saved to {output_dir}")
    
    def predict_churn_risk(self, data, customer_ids):
        """Predict churn risk for customers"""
        if self.model is None:
            print("Model not trained yet")
            return None
        
        # Make predictions
        churn_probabilities = self.model.predict_proba(data)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': churn_probabilities,
            'churn_risk': pd.cut(churn_probabilities, 
                               bins=[0, 0.3, 0.7, 1.0], 
                               labels=['Low', 'Medium', 'High'])
        })
        
        # Sort by churn probability (highest risk first)
        results = results.sort_values('churn_probability', ascending=False)
        
        return results
    
    def train_and_evaluate(self, data_path='data/featured_data.csv', tune_hyperparameters=True):
        """Complete training and evaluation pipeline"""
        
        # Load data
        data = self.load_data(data_path)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        # Train model
        self.train_random_forest(X_train, y_train, tune_hyperparameters)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        # Cross-validation
        cv_scores = self.cross_validate_model(X_train, y_train)
        
        # Feature importance
        feature_names = X_train.columns
        self.get_feature_importance(feature_names)
        
        # Create plots
        self.plot_feature_importance()
        self.plot_roc_curve(y_test, evaluation_results['y_pred_proba'])
        self.plot_precision_recall_curve(y_test, evaluation_results['y_pred_proba'])
        
        # Save model
        self.save_model()
        
        # Predict churn risk for all customers
        customer_ids = data['customer_id']
        X_all = data.drop(['customer_id', 'churned'], axis=1)
        churn_risk_predictions = self.predict_churn_risk(X_all, customer_ids)
        
        # Save predictions
        churn_risk_predictions.to_csv('data/churn_risk_predictions.csv', index=False)
        print(f"Churn risk predictions saved to data/churn_risk_predictions.csv")
        
        return evaluation_results, churn_risk_predictions

def main():
    """Main model training execution"""
    
    # Initialize model
    model = ChurnPredictionModel()
    
    # Train and evaluate
    print("\n" + "="*60)
    print("🚀 STARTING MODEL TRAINING AND EVALUATION")
    print("="*60)
    #evaluation_results, churn_predictions = model.train_and_evaluate(tune_hyperparameters=False)
    evaluation_results, churn_predictions = model.train_and_evaluate(tune_hyperparameters=True)
    
    print("\n=== Model Training Complete ===")
    print(f"Model accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Model F1-score: {evaluation_results['f1_score']:.4f}")
    print(f"Model ROC-AUC: {evaluation_results['roc_auc']:.4f}")
    
    # Show top 10 high-risk customers
    print("\nTop 10 High-Risk Customers:")
    print(churn_predictions.head(10))

if __name__ == "__main__":
    main()
