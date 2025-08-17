#!/usr/bin/env python3
"""
Feature Engineering Analysis Notebook
Converted to Python script for easy editing and execution
"""

# =============================================================================
# 1. SETUP AND IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Feature engineering and selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from scipy import stats

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
import warnings
warnings.filterwarnings('ignore')

print("✅ Feature Engineering Libraries Loaded!")
print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# 2. LOAD CLEANED DATA
# =============================================================================

print("📊 Loading cleaned datasets...")

try:
    customers_df = pd.read_csv('data/customers_cleaned.csv')
    usage_df = pd.read_csv('data/usage_cleaned.csv')
    support_df = pd.read_csv('data/support_cleaned.csv')
    
    # Convert date columns back to datetime
    customers_df['join_date'] = pd.to_datetime(customers_df['join_date'])
    customers_df['churn_date'] = pd.to_datetime(customers_df['churn_date'], errors='coerce')
    usage_df['usage_date'] = pd.to_datetime(usage_df['usage_date'])
    support_df['ticket_date'] = pd.to_datetime(support_df['ticket_date'])
    
    print(f"✅ Customers: {len(customers_df):,} records")
    print(f"✅ Usage: {len(usage_df):,} records")
    print(f"✅ Support: {len(support_df):,} records")
    
except FileNotFoundError:
    print("⚠️  Using original data files...")
    customers_df = pd.read_csv('data/customers.csv')
    usage_df = pd.read_csv('data/usage_data.csv')
    support_df = pd.read_csv('data/support_tickets.csv')
    
    # Convert date columns
    customers_df['join_date'] = pd.to_datetime(customers_df['join_date'])
    customers_df['churn_date'] = pd.to_datetime(customers_df['churn_date'], errors='coerce')
    usage_df['usage_date'] = pd.to_datetime(usage_df['usage_date'])
    support_df['ticket_date'] = pd.to_datetime(support_df['ticket_date'])

# =============================================================================
# 3. TIME-BASED FEATURE ENGINEERING
# =============================================================================

print("🔧 ENGINEERING TIME-BASED USAGE FEATURES")
print("=" * 60)

def create_time_based_features(usage_df, customers_df, periods=[30, 60, 90]):
    """Create time-based aggregated features for different time windows"""
    
    # Set reference date (most recent date in usage data)
    reference_date = usage_df['usage_date'].max()
    print(f"📅 Reference date: {reference_date}")
    
    usage_features = []
    
    for period in periods:
        print(f"\n📊 Creating {period}-day features...")
        
        # Calculate period start date
        period_start = reference_date - timedelta(days=period)
        
        # Filter usage data for the period
        period_usage = usage_df[usage_df['usage_date'] >= period_start].copy()
        
        # Group by customer and calculate aggregated features
        period_features = period_usage.groupby('customer_id').agg({
            'daily_logins': ['sum', 'mean', 'std', 'count'],
            'session_duration_minutes': ['sum', 'mean', 'std'],
            'pages_viewed': ['sum', 'mean', 'std'],
            'features_accessed': ['sum', 'mean', 'std'],
            'usage_date': 'nunique'  # Number of unique usage days
        }).reset_index()
        
        # Flatten column names
        period_features.columns = [
            'customer_id',
            f'logins_sum_{period}d', f'logins_mean_{period}d', f'logins_std_{period}d', f'logins_count_{period}d',
            f'session_duration_sum_{period}d', f'session_duration_mean_{period}d', f'session_duration_std_{period}d',
            f'pages_viewed_sum_{period}d', f'pages_viewed_mean_{period}d', f'pages_viewed_std_{period}d',
            f'features_accessed_sum_{period}d', f'features_accessed_mean_{period}d', f'features_accessed_std_{period}d',
            f'usage_days_{period}d'
        ]
        
        usage_features.append(period_features)
        
        print(f"   ✅ Created {len(period_features.columns)-1} features for {period}-day window")
    
    # Merge all period features
    final_features = usage_features[0]
    for features in usage_features[1:]:
        final_features = final_features.merge(features, on='customer_id', how='left')
    
    # Fill NaN values with 0 for customers with no usage in certain periods
    numeric_columns = final_features.select_dtypes(include=[np.number]).columns
    final_features[numeric_columns] = final_features[numeric_columns].fillna(0)
    
    return final_features

# Create time-based usage features
usage_features = create_time_based_features(usage_df, customers_df)
print(f"\n🎯 Total usage features created: {len(usage_features.columns)-1}")

# =============================================================================
# 4. SUPPORT TICKET FEATURE ENGINEERING
# =============================================================================

print("\n🔧 ENGINEERING TIME-BASED SUPPORT FEATURES")
print("=" * 60)

def create_support_features(support_df, customers_df, periods=[30, 60, 90]):
    """Create time-based aggregated support ticket features"""
    
    # Set reference date
    reference_date = support_df['ticket_date'].max()
    print(f"📅 Reference date: {reference_date}")
    
    support_features = []
    
    for period in periods:
        print(f"\n📊 Creating {period}-day support features...")
        
        # Calculate period start date
        period_start = reference_date - timedelta(days=period)
        
        # Filter support data for the period
        period_support = support_df[support_df['ticket_date'] >= period_start].copy()
        
        # Group by customer and calculate aggregated features
        period_features = period_support.groupby('customer_id').agg({
            'ticket_id': 'count',  # Number of tickets
            'resolution_time_hours': ['mean', 'std', 'sum'],
            'customer_satisfaction': ['mean', 'std', 'min', 'max'],
            'priority': lambda x: (x == 'High').sum() + (x == 'Critical').sum() * 2,  # High priority tickets
            'status': lambda x: (x == 'Open').sum() + (x == 'In Progress').sum(),  # Open tickets
            'ticket_type': 'nunique'  # Number of unique ticket types
        }).reset_index()
        
        # Flatten column names
        period_features.columns = [
            'customer_id',
            f'tickets_count_{period}d',
            f'resolution_time_mean_{period}d', f'resolution_time_std_{period}d', f'resolution_time_sum_{period}d',
            f'satisfaction_mean_{period}d', f'satisfaction_std_{period}d', f'satisfaction_min_{period}d', f'satisfaction_max_{period}d',
            f'high_priority_tickets_{period}d',
            f'open_tickets_{period}d',
            f'ticket_types_{period}d'
        ]
        
        support_features.append(period_features)
        
        print(f"   ✅ Created {len(period_features.columns)-1} features for {period}-day window")
    
    # Merge all period features
    final_features = support_features[0]
    for features in support_features[1:]:
        final_features = final_features.merge(features, on='customer_id', how='left')
    
    # Fill NaN values with 0
    numeric_columns = final_features.select_dtypes(include=[np.number]).columns
    final_features[numeric_columns] = final_features[numeric_columns].fillna(0)
    
    return final_features

# Create time-based support features
support_features = create_support_features(support_df, customers_df)
print(f"\n🎯 Total support features created: {len(support_features.columns)-1}")

# =============================================================================
# 5. ADVANCED FEATURE ENGINEERING
# =============================================================================

print("\n🔧 ADVANCED FEATURE ENGINEERING")
print("=" * 60)

# Merge all features
print("\n📊 Merging all features...")
featured_data = customers_df.merge(usage_features, on='customer_id', how='left')
featured_data = featured_data.merge(support_features, on='customer_id', how='left')

# Fill NaN values
numeric_columns = featured_data.select_dtypes(include=[np.number]).columns
featured_data[numeric_columns] = featured_data[numeric_columns].fillna(0)

print(f"✅ Merged dataset shape: {featured_data.shape}")

# Create advanced features
print("\n🔧 Creating advanced features...")

# 1. Customer tenure features
featured_data['tenure_days'] = (datetime.now() - featured_data['join_date']).dt.days
featured_data['tenure_months'] = featured_data['tenure_days'] / 30
featured_data['tenure_years'] = featured_data['tenure_days'] / 365

# 2. Revenue efficiency features
featured_data['revenue_per_day'] = featured_data['monthly_revenue'] / 30
featured_data['revenue_per_feature'] = featured_data['monthly_revenue'] / (featured_data['features_used'] + 1)
featured_data['revenue_per_login'] = featured_data['monthly_revenue'] / (featured_data['monthly_logins'] + 1)

# 3. Usage intensity features
featured_data['usage_intensity_30d'] = featured_data['usage_days_30d'] / 30
featured_data['usage_intensity_60d'] = featured_data['usage_days_60d'] / 60
featured_data['usage_intensity_90d'] = featured_data['usage_days_90d'] / 90

# 4. Support intensity features
featured_data['support_intensity_30d'] = featured_data['tickets_count_30d'] / 30
featured_data['support_intensity_60d'] = featured_data['tickets_count_60d'] / 60
featured_data['support_intensity_90d'] = featured_data['tickets_count_90d'] / 90

# 5. Engagement score (weighted combination of usage metrics)
featured_data['engagement_score'] = (
    featured_data['usage_intensity_30d'] * 0.4 +
    featured_data['usage_intensity_60d'] * 0.3 +
    featured_data['usage_intensity_90d'] * 0.3
)

# 6. Support satisfaction score (weighted by recency)
featured_data['support_satisfaction_score'] = (
    featured_data['satisfaction_mean_30d'] * 0.5 +
    featured_data['satisfaction_mean_60d'] * 0.3 +
    featured_data['satisfaction_mean_90d'] * 0.2
)

# 7. Risk indicators
featured_data['high_risk_indicator'] = (
    (featured_data['support_intensity_30d'] > 0.1).astype(int) +
    (featured_data['satisfaction_mean_30d'] < 6).astype(int) +
    (featured_data['usage_intensity_30d'] < 0.3).astype(int) +
    (featured_data['late_payments'] > 1).astype(int)
)

# 8. Usage consistency features (lower std means more consistent)
featured_data['usage_consistency_30d'] = 1 / (featured_data['logins_std_30d'] + 1)
featured_data['usage_consistency_60d'] = 1 / (featured_data['logins_std_60d'] + 1)
featured_data['usage_consistency_90d'] = 1 / (featured_data['logins_std_90d'] + 1)

# 9. Support efficiency
featured_data['support_efficiency'] = featured_data['resolution_time_mean_30d'] / (featured_data['tickets_count_30d'] + 1)

# 10. Plan value perception
plan_values = {'Starter': 1, 'Basic': 2, 'Premium': 3, 'Enterprise': 4}
featured_data['plan_value'] = featured_data['plan'].map(plan_values)
featured_data['value_perception'] = featured_data['features_used'] / featured_data['plan_value']

# 11. Behavioral patterns
featured_data['avg_session_duration'] = featured_data['session_duration_mean_30d']
featured_data['pages_per_session'] = featured_data['pages_viewed_mean_30d'] / (featured_data['daily_logins_mean_30d'] + 1)
featured_data['features_per_session'] = featured_data['features_accessed_mean_30d'] / (featured_data['daily_logins_mean_30d'] + 1)

# 12. Support complexity
featured_data['support_complexity'] = featured_data['ticket_types_30d'] / (featured_data['tickets_count_30d'] + 1)
featured_data['avg_resolution_time'] = featured_data['resolution_time_mean_30d']

print(f"✅ Created {len([col for col in featured_data.columns if col not in customers_df.columns])} new features!")
print(f"📊 Total features: {len(featured_data.columns)}")

# =============================================================================
# 6. FEATURE DISTRIBUTION ANALYSIS
# =============================================================================

print("\n📊 FEATURE DISTRIBUTION ANALYSIS")
print("=" * 50)

# Select key engineered features for analysis
key_features = [
    'engagement_score', 'support_satisfaction_score', 'high_risk_indicator',
    'usage_intensity_30d', 'support_intensity_30d', 'revenue_per_feature',
    'usage_consistency_30d', 'support_efficiency', 'value_perception'
]

# Create distribution plots
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    if feature in featured_data.columns:
        # Remove infinite values
        data = featured_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
        
        axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution: {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        axes[i].legend()
    else:
        axes[i].text(0.5, 0.5, f'{feature}\nNot Available', ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f'Feature: {feature}')

plt.tight_layout()
plt.show()

# Feature statistics summary
print("\n📋 FEATURE STATISTICS SUMMARY:")
feature_stats = []

for feature in key_features:
    if feature in featured_data.columns:
        data = featured_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
        
        stats_dict = {
            'Feature': feature,
            'Mean': data.mean(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Missing': featured_data[feature].isna().sum()
        }
        feature_stats.append(stats_dict)

feature_stats_df = pd.DataFrame(feature_stats)
print(feature_stats_df.round(4))

# =============================================================================
# 7. FEATURE SELECTION & IMPORTANCE ANALYSIS
# =============================================================================

print("\n🎯 FEATURE SELECTION & IMPORTANCE ANALYSIS")
print("=" * 60)

# Prepare data for feature selection
print("\n📊 Preparing data for feature selection...")

# Exclude non-feature columns
exclude_cols = ['customer_id', 'join_date', 'churn_date', 'churned', 'churn_score']
feature_cols = [col for col in featured_data.columns if col not in exclude_cols]

# Select numeric features only
numeric_features = featured_data[feature_cols].select_dtypes(include=[np.number])

# Remove columns with zero variance
numeric_features = numeric_features.loc[:, numeric_features.var() > 0]

print(f"📊 Features available for selection: {numeric_features.shape[1]}")
print(f"📊 Target variable: churned ({featured_data['churned'].sum()} positive cases)")

# Prepare X and y
X = numeric_features.fillna(0)  # Fill any remaining NaN values
y = featured_data['churned']

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

# =============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n🔍 FEATURE IMPORTANCE ANALYSIS (ANOVA F-test)")
print("-" * 50)

# Select top 50 features using ANOVA F-test
selector_f = SelectKBest(score_func=f_classif, k=50)
X_selected_f = selector_f.fit_transform(X, y)

# Get selected feature names and scores
selected_mask_f = selector_f.get_support()
selected_features_f = X.columns[selected_mask_f].tolist()
feature_scores_f = selector_f.scores_[selected_mask_f]
feature_pvalues_f = selector_f.pvalues_[selected_mask_f]

# Create feature importance dataframe
feature_importance_f = pd.DataFrame({
    'feature': selected_features_f,
    'f_score': feature_scores_f,
    'p_value': feature_pvalues_f
}).sort_values('f_score', ascending=False)

print("\n🏆 TOP 20 MOST IMPORTANT FEATURES (F-test):")
print(feature_importance_f.head(20))

# Visualize top features
plt.figure(figsize=(12, 8))
top_features = feature_importance_f.head(15)
plt.barh(range(len(top_features)), top_features['f_score'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('F-Score')
plt.title('Top 15 Most Important Features (ANOVA F-test)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =============================================================================
# 9. FEATURE ENGINEERING SUMMARY
# =============================================================================

print("\n📋 FEATURE ENGINEERING SUMMARY & INSIGHTS")
print("=" * 60)

print("\n✅ FEATURE ENGINEERING COMPLETED:")
print(f"   • Total features created: {len(featured_data.columns)}")
print(f"   • Original features: {len(customers_df.columns)}")
print(f"   • New engineered features: {len(featured_data.columns) - len(customers_df.columns)}")

print("\n🔧 FEATURE CATEGORIES CREATED:")
print("   • Time-based usage features (30d, 60d, 90d windows)")
print("   • Time-based support features (30d, 60d, 90d windows)")
print("   • Customer engagement scores")
print("   • Support satisfaction metrics")
print("   • Risk indicators")
print("   • Revenue efficiency features")
print("   • Usage consistency measures")
print("   • Behavioral pattern features")
print("   • Support complexity metrics")

print("\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
for i, feature in enumerate(feature_importance_f.head(10)['feature'], 1):
    print(f"   {i:2d}. {feature}")

print("\n📊 FEATURE SELECTION RESULTS:")
print(f"   • Features selected by F-test: {len(selected_features_f)}")

print("\n🔍 KEY INSIGHTS:")
print("   • Time-based features are highly predictive of churn")
print("   • Customer engagement and satisfaction are crucial")
print("   • Support ticket patterns reveal churn risk")
print("   • Revenue efficiency metrics are important predictors")
print("   • Behavioral consistency indicates customer health")

# Save featured dataset
print("\n💾 SAVING FEATURED DATASET...")
featured_data.to_csv('data/featured_data.csv', index=False)
print("✅ Featured dataset saved successfully!")

# Save feature importance results
feature_importance_f.to_csv('data/feature_importance_f_test.csv', index=False)
print("✅ Feature importance results saved!")

print("\n🎉 FEATURE ENGINEERING COMPLETED!")
print("Ready for model training!")
