#!/usr/bin/env python3
"""
Model Training and Analysis Notebook
Converted to Python script for easy editing and execution
"""

# =============================================================================
# 1. SETUP AND IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, f_classif

# Additional libraries
import joblib
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Model Training Libraries Loaded!")
print(f"📅 Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# 2. LOAD FEATURED DATA
# =============================================================================

print("📊 Loading featured dataset...")

try:
    featured_data = pd.read_csv('data/featured_data.csv')
    print(f"✅ Featured data loaded: {featured_data.shape}")
except FileNotFoundError:
    print("⚠️  Featured data not found. Loading processed data...")
    featured_data = pd.read_csv('data/processed_data.csv')
    print(f"✅ Processed data loaded: {featured_data.shape}")

# Display basic information
print(f"\n📋 Dataset Overview:")
print(f"   • Total records: {len(featured_data):,}")
print(f"   • Total features: {len(featured_data.columns)}")
print(f"   • Churn rate: {featured_data['churned'].mean():.2%}")
print(f"   • Churned customers: {featured_data['churned'].sum():,}")
print(f"   • Non-churned customers: {(featured_data['churned'] == 0).sum():,}")

# =============================================================================
# 3. DATA PREPARATION FOR MODELING
# =============================================================================

print("\n🔧 PREPARING DATA FOR MODELING")
print("=" * 50)

# Identify categorical and numerical columns
categorical_columns = featured_data.select_dtypes(include=['object']).columns.tolist()
numerical_columns = featured_data.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n📊 Column Analysis:")
print(f"   • Categorical columns: {len(categorical_columns)}")
print(f"   • Numerical columns: {len(numerical_columns)}")

if categorical_columns:
    print(f"   • Categorical columns: {categorical_columns}")

# Prepare features and target
print("\n🎯 Preparing features and target variable...")

# Exclude non-feature columns
exclude_cols = ['customer_id', 'join_date', 'churn_date', 'churned', 'churn_score']
feature_cols = [col for col in featured_data.columns if col not in exclude_cols]

# Select features
X = featured_data[feature_cols].copy()
y = featured_data['churned']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"✅ Target distribution: {y.value_counts().to_dict()}")

# =============================================================================
# 4. HANDLE CATEGORICAL FEATURES
# =============================================================================

print("\n🔧 Handling categorical features...")

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

if categorical_features:
    print(f"   • Categorical features found: {categorical_features}")
    
    # Encode categorical features
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"   ✅ Encoded {col}: {len(le.classes_)} unique values")
else:
    print("   • No categorical features found")
    X_encoded = X.copy()

print(f"✅ Encoded features shape: {X_encoded.shape}")

# =============================================================================
# 5. HANDLE MISSING VALUES
# =============================================================================

print("\n🔧 Handling missing values...")

missing_counts = X_encoded.isnull().sum()
missing_features = missing_counts[missing_counts > 0]

if len(missing_features) > 0:
    print(f"   • Features with missing values: {len(missing_features)}")
    print("   • Missing value counts:")
    for feature, count in missing_features.items():
        print(f"     - {feature}: {count} ({count/len(X_encoded)*100:.2f}%)")
    
    # Fill missing values
    X_encoded = X_encoded.fillna(0)  # Fill with 0 for numerical features
    print("   ✅ Missing values filled with 0")
else:
    print("   ✅ No missing values found")

# Remove infinite values
infinite_counts = np.isinf(X_encoded.select_dtypes(include=[np.number])).sum()
infinite_features = infinite_counts[infinite_counts > 0]

if len(infinite_features) > 0:
    print(f"   • Features with infinite values: {len(infinite_features)}")
    # Replace infinite values with large numbers
    X_encoded = X_encoded.replace([np.inf, -np.inf], 0)
    print("   ✅ Infinite values replaced with 0")
else:
    print("   ✅ No infinite values found")

print(f"✅ Final features shape: {X_encoded.shape}")

# =============================================================================
# 6. FEATURE SELECTION
# =============================================================================

print("\n🎯 Performing feature selection...")

# Select top 50 features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=50)
X_selected = selector.fit_transform(X_encoded, y)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = X_encoded.columns[selected_mask].tolist()
feature_scores = selector.scores_[selected_mask]

print(f"✅ Selected {len(selected_features)} features")

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'f_score': feature_scores
}).sort_values('f_score', ascending=False)

print("\n🏆 TOP 10 SELECTED FEATURES:")
print(feature_importance.head(10))

# Update X with selected features
X_final = X_encoded[selected_features]
print(f"✅ Final feature matrix shape: {X_final.shape}")

# =============================================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING SETS
# =============================================================================

print("\n📊 Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {X_train.shape} ({len(X_train)/len(X_final)*100:.1f}%)")
print(f"✅ Testing set: {X_test.shape} ({len(X_test)/len(X_final)*100:.1f}%)")
print(f"✅ Training churn rate: {y_train.mean():.2%}")
print(f"✅ Testing churn rate: {y_test.mean():.2%}")

# =============================================================================
# 8. RANDOM FOREST MODEL TRAINING
# =============================================================================

print("\n🤖 RANDOM FOREST MODEL TRAINING")
print("=" * 50)

# Initialize Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("\n🔧 Model Parameters:")
print(f"   • n_estimators: {rf_model.n_estimators}")
print(f"   • max_depth: {rf_model.max_depth}")
print(f"   • min_samples_split: {rf_model.min_samples_split}")
print(f"   • min_samples_leaf: {rf_model.min_samples_leaf}")
print(f"   • max_features: {rf_model.max_features}")
print(f"   • class_weight: {rf_model.class_weight}")

# Train the model
print("\n🚀 Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("✅ Model training completed!")

# =============================================================================
# 9. MAKE PREDICTIONS
# =============================================================================

print("\n🎯 Making predictions...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"✅ Predictions completed!")
print(f"   • Predicted churn rate: {y_pred.mean():.2%}")
print(f"   • Actual churn rate: {y_test.mean():.2%}")

# =============================================================================
# 10. MODEL PERFORMANCE EVALUATION
# =============================================================================

print("\n📊 MODEL PERFORMANCE EVALUATION")
print("=" * 50)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n🏆 PERFORMANCE METRICS:")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   • Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"   • F1-Score: {f1:.4f} ({f1*100:.2f}%)")
print(f"   • ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# Check if we meet the 85% accuracy target from resume
if accuracy >= 0.85:
    print(f"\n🎉 SUCCESS: Model accuracy ({accuracy*100:.2f}%) exceeds target (85%)!")
else:
    print(f"\n⚠️  Model accuracy ({accuracy*100:.2f}%) below target (85%)")

# =============================================================================
# 11. CONFUSION MATRIX
# =============================================================================

print("\n📊 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(f"\n📋 Confusion Matrix Details:")
print(f"   • True Negatives: {cm[0, 0]} (Correctly predicted not churned)")
print(f"   • False Positives: {cm[0, 1]} (Incorrectly predicted churned)")
print(f"   • False Negatives: {cm[1, 0]} (Incorrectly predicted not churned)")
print(f"   • True Positives: {cm[1, 1]} (Correctly predicted churned)")

# =============================================================================
# 12. CLASSIFICATION REPORT
# =============================================================================

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))

# =============================================================================
# 13. ROC CURVE
# =============================================================================

print("\n📊 ROC CURVE:")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 14. PRECISION-RECALL CURVE
# =============================================================================

print("\n📊 PRECISION-RECALL CURVE:")
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'Precision-Recall Curve')
plt.axhline(y=y_test.mean(), color='red', linestyle='--', label=f'Baseline (Churn Rate = {y_test.mean():.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 15. CROSS-VALIDATION ANALYSIS
# =============================================================================

print("\n🔄 CROSS-VALIDATION ANALYSIS")
print("=" * 50)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X_final, y, cv=5, scoring='f1')

print(f"\n📊 Cross-Validation Results (F1-Score):")
print(f"   • CV Scores: {cv_scores}")
print(f"   • Mean CV Score: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"   • CV Score Std: {cv_scores.std():.4f} ({cv_scores.std()*100:.2f}%)")
print(f"   • CV Score Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

# Visualize cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), cv_scores, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean CV Score: {cv_scores.mean():.3f}')
plt.fill_between(range(1, 6), cv_scores.mean() - cv_scores.std(), 
                 cv_scores.mean() + cv_scores.std(), alpha=0.2, color='red')
plt.xlabel('Fold')
plt.ylabel('F1-Score')
plt.title('5-Fold Cross-Validation Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 16. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

# Get feature importance from Random Forest
rf_importance = rf_model.feature_importances_
feature_names = selected_features

# Create feature importance dataframe
rf_feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_importance
}).sort_values('importance', ascending=False)

print("\n🏆 TOP 20 MOST IMPORTANT FEATURES (Random Forest):")
print(rf_feature_importance.head(20))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = rf_feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =============================================================================
# 17. COMPARE FEATURE IMPORTANCE METHODS
# =============================================================================

print("\n📊 COMPARING FEATURE IMPORTANCE METHODS")
print("=" * 50)

# Merge F-test and Random Forest importance
comparison_df = feature_importance.merge(rf_feature_importance, on='feature', how='inner')
comparison_df['f_score_normalized'] = comparison_df['f_score'] / comparison_df['f_score'].max()
comparison_df['importance_normalized'] = comparison_df['importance'] / comparison_df['importance'].max()
comparison_df['avg_importance'] = (comparison_df['f_score_normalized'] + comparison_df['importance_normalized']) / 2

comparison_df = comparison_df.sort_values('avg_importance', ascending=False)

print("\n🏆 TOP 15 FEATURES BY COMBINED IMPORTANCE:")
print(comparison_df.head(15))

# Visualize comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot
ax1.scatter(comparison_df['f_score_normalized'], comparison_df['importance_normalized'], alpha=0.7)
ax1.set_xlabel('F-Score (Normalized)')
ax1.set_ylabel('Random Forest Importance (Normalized)')
ax1.set_title('Feature Importance: F-Score vs Random Forest')
ax1.grid(True, alpha=0.3)

# Top features by combined importance
top_combined = comparison_df.head(10)
ax2.barh(range(len(top_combined)), top_combined['avg_importance'])
ax2.set_yticks(range(len(top_combined)))
ax2.set_yticklabels(top_combined['feature'])
ax2.set_xlabel('Average Importance')
ax2.set_title('Top 10 Features by Combined Importance')
ax2.invert_yaxis()

plt.tight_layout()
plt.show()

# =============================================================================
# 18. GENERATE CHURN PREDICTIONS
# =============================================================================

print("\n🎯 GENERATING CHURN PREDICTIONS")
print("=" * 50)

# Make predictions on full dataset
all_predictions = rf_model.predict(X_final)
all_probabilities = rf_model.predict_proba(X_final)[:, 1]

# Create predictions dataframe
predictions_df = featured_data[['customer_id']].copy()
predictions_df['churn_probability'] = all_probabilities
predictions_df['predicted_churn'] = all_predictions
predictions_df['actual_churn'] = y

# Add risk categories
def categorize_risk(prob):
    if prob < 0.3:
        return 'Low'
    elif prob < 0.7:
        return 'Medium'
    else:
        return 'High'

predictions_df['risk_category'] = predictions_df['churn_probability'].apply(categorize_risk)

print(f"\n📊 Prediction Summary:")
print(f"   • Total customers: {len(predictions_df):,}")
print(f"   • Predicted to churn: {predictions_df['predicted_churn'].sum():,} ({predictions_df['predicted_churn'].mean():.2%})")
print(f"   • Actually churned: {predictions_df['actual_churn'].sum():,} ({predictions_df['actual_churn'].mean():.2%})")

print(f"\n🎯 Risk Category Distribution:")
risk_dist = predictions_df['risk_category'].value_counts()
for category, count in risk_dist.items():
    print(f"   • {category} Risk: {count:,} ({count/len(predictions_df)*100:.1f}%)")

# Identify top 15% at-risk customers
top_15_percent = int(len(predictions_df) * 0.15)
high_risk_customers = predictions_df.nlargest(top_15_percent, 'churn_probability')

print(f"\n🎯 Top 15% At-Risk Customers:")
print(f"   • Count: {len(high_risk_customers):,}")
print(f"   • Average churn probability: {high_risk_customers['churn_probability'].mean():.3f}")
print(f"   • Actually churned: {high_risk_customers['actual_churn'].sum():,} ({high_risk_customers['actual_churn'].mean():.2%})")

# Display sample of high-risk customers
print(f"\n📋 Sample of High-Risk Customers:")
print(high_risk_customers.head(10))

# =============================================================================
# 19. MODEL SUMMARY & BUSINESS IMPACT
# =============================================================================

print("\n📋 MODEL SUMMARY & BUSINESS IMPACT")
print("=" * 60)

print("\n✅ MODEL PERFORMANCE SUMMARY:")
print(f"   • Model Type: Random Forest Classifier")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   • Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"   • F1-Score: {f1:.4f} ({f1*100:.2f}%)")
print(f"   • ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print(f"   • Cross-Validation F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n🎯 BUSINESS IMPACT:")
print(f"   • Total customers analyzed: {len(predictions_df):,}")
print(f"   • High-risk customers identified: {len(high_risk_customers):,} (15%)")
print(f"   • Model accuracy exceeds 85% target: {'✅ YES' if accuracy >= 0.85 else '❌ NO'}")
print(f"   • Ready for retention team targeting")

print("\n🔧 TECHNICAL DETAILS:")
print(f"   • Features used: {len(selected_features)}")
print(f"   • Training samples: {len(X_train):,}")
print(f"   • Testing samples: {len(X_test):,}")
print(f"   • Model training completed successfully")

# =============================================================================
# 20. SAVE MODEL AND PREDICTIONS
# =============================================================================

print("\n💾 SAVING MODEL AND PREDICTIONS...")

# Save the trained model
joblib.dump(rf_model, 'models/random_forest_model.pkl')
print("✅ Model saved: models/random_forest_model.pkl")

# Save predictions
predictions_df.to_csv('data/churn_risk_predictions.csv', index=False)
print("✅ Predictions saved: data/churn_risk_predictions.csv")

# Save feature importance
rf_feature_importance.to_csv('data/feature_importance.csv', index=False)
print("✅ Feature importance saved: data/feature_importance.csv")

# Save selected features
pd.DataFrame({'feature': selected_features}).to_csv('data/selected_features.csv', index=False)
print("✅ Selected features saved: data/selected_features.csv")

print("\n🎉 MODEL TRAINING AND ANALYSIS COMPLETED!")
print("Ready for dashboard implementation and business deployment!")
