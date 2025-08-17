"""
Sample Data Generator for Churn Prediction
Creates realistic customer data with usage patterns, support tickets, and churn indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3
import os

def generate_customer_data(n_customers=10000):
    """Generate realistic customer data"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Customer demographics
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    
    # Subscription plans
    plans = ['Basic', 'Premium', 'Enterprise', 'Starter']
    plan_weights = [0.4, 0.3, 0.2, 0.1]
    
    # Geographic data
    countries = ['US', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan']
    country_weights = [0.5, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]
    
    # Industry sectors
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Education', 'Manufacturing']
    industry_weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
    
    data = []
    
    for i, customer_id in enumerate(customer_ids):
        # Basic customer info
        join_date = datetime.now() - timedelta(days=random.randint(30, 1095))  # 1 month to 3 years
        plan = np.random.choice(plans, p=plan_weights)
        country = np.random.choice(countries, p=country_weights)
        industry = np.random.choice(industries, p=industry_weights)
        
        # Monthly revenue (varies by plan)
        plan_revenue = {'Basic': 29, 'Premium': 79, 'Enterprise': 199, 'Starter': 9}
        base_revenue = plan_revenue[plan]
        monthly_revenue = base_revenue + np.random.normal(0, base_revenue * 0.2)
        
        # Usage metrics
        monthly_logins = np.random.poisson(15) + np.random.normal(0, 5)
        features_used = np.random.randint(3, 15)
        data_usage_gb = np.random.exponential(50)
        
        # Support interactions
        support_tickets = np.random.poisson(2)
        avg_response_time = np.random.exponential(24)  # hours
        satisfaction_score = np.random.normal(7.5, 1.5)
        satisfaction_score = max(1, min(10, satisfaction_score))
        
        # Payment behavior
        payment_method = np.random.choice(['Credit Card', 'Bank Transfer', 'PayPal'], p=[0.7, 0.2, 0.1])
        late_payments = np.random.poisson(0.5)
        
        # Churn indicators (will be used to create target variable)
        churn_risk_factors = []
        
        # High churn risk factors
        if monthly_logins < 5:
            churn_risk_factors.append(3)
        if support_tickets > 5:
            churn_risk_factors.append(2)
        if satisfaction_score < 5:
            churn_risk_factors.append(3)
        if late_payments > 2:
            churn_risk_factors.append(2)
        if data_usage_gb < 10:
            churn_risk_factors.append(1)
        
        # Low churn risk factors
        if monthly_logins > 25:
            churn_risk_factors.append(-2)
        if satisfaction_score > 8:
            churn_risk_factors.append(-2)
        if features_used > 10:
            churn_risk_factors.append(-1)
        
        # Calculate churn probability
        churn_score = sum(churn_risk_factors) + np.random.normal(0, 1)
        churned = 1 if churn_score > 2 else 0
        
        # Churn date (if churned)
        churn_date = None
        if churned:
            churn_date = join_date + timedelta(days=random.randint(30, 365))
        
        data.append({
            'customer_id': customer_id,
            'join_date': join_date,
            'plan': plan,
            'country': country,
            'industry': industry,
            'monthly_revenue': round(monthly_revenue, 2),
            'monthly_logins': max(0, int(monthly_logins)),
            'features_used': features_used,
            'data_usage_gb': round(data_usage_gb, 2),
            'support_tickets': support_tickets,
            'avg_response_time_hours': round(avg_response_time, 1),
            'satisfaction_score': round(satisfaction_score, 1),
            'payment_method': payment_method,
            'late_payments': late_payments,
            'churned': churned,
            'churn_date': churn_date,
            'churn_score': round(churn_score, 2)
        })
    
    return pd.DataFrame(data)

def generate_usage_data(customer_data, n_records=50000):
    """Generate daily usage data for customers"""
    
    usage_data = []
    
    for _, customer in customer_data.iterrows():
        # Generate usage records for the last 90 days
        start_date = datetime.now() - timedelta(days=90)
        
        # Number of usage days (varies by customer)
        usage_days = np.random.poisson(45)  # Average 45 days out of 90
        
        for _ in range(usage_days):
            usage_date = start_date + timedelta(days=random.randint(0, 90))
            
            # Usage metrics for this day
            daily_logins = np.random.poisson(2) + 1
            session_duration_minutes = np.random.exponential(30)
            pages_viewed = np.random.poisson(15)
            features_accessed = np.random.randint(1, 8)
            
            usage_data.append({
                'customer_id': customer['customer_id'],
                'usage_date': usage_date,
                'daily_logins': daily_logins,
                'session_duration_minutes': round(session_duration_minutes, 1),
                'pages_viewed': pages_viewed,
                'features_accessed': features_accessed
            })
    
    return pd.DataFrame(usage_data)

def generate_support_tickets(customer_data, n_tickets=15000):
    """Generate support ticket data"""
    
    ticket_data = []
    ticket_types = ['Technical Issue', 'Billing Question', 'Feature Request', 'Account Access', 'General Inquiry']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    statuses = ['Open', 'In Progress', 'Resolved', 'Closed']
    
    for _, customer in customer_data.iterrows():
        # Number of tickets for this customer
        n_customer_tickets = np.random.poisson(customer['support_tickets'])
        
        for _ in range(n_customer_tickets):
            ticket_date = customer['join_date'] + timedelta(days=random.randint(0, 365))
            
            ticket_type = random.choice(ticket_types)
            priority = random.choice(priorities)
            status = random.choice(statuses)
            
            # Resolution time based on priority
            resolution_times = {'Low': 72, 'Medium': 48, 'High': 24, 'Critical': 4}
            avg_resolution = resolution_times[priority]
            resolution_time = np.random.exponential(avg_resolution)
            
            ticket_data.append({
                'customer_id': customer['customer_id'],
                'ticket_id': f"TKT_{len(ticket_data):06d}",
                'ticket_date': ticket_date,
                'ticket_type': ticket_type,
                'priority': priority,
                'status': status,
                'resolution_time_hours': round(resolution_time, 1),
                'customer_satisfaction': np.random.normal(7, 2)
            })
    
    return pd.DataFrame(ticket_data)

def create_database():
    """Create SQLite database and populate with sample data"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    print("Generating customer data...")
    customer_data = generate_customer_data(10000)
    
    print("Generating usage data...")
    usage_data = generate_usage_data(customer_data, 50000)
    
    print("Generating support ticket data...")
    ticket_data = generate_support_tickets(customer_data, 15000)
    
    # Save to SQLite database
    conn = sqlite3.connect('data/churn_prediction.db')
    
    customer_data.to_sql('customers', conn, if_exists='replace', index=False)
    usage_data.to_sql('usage_data', conn, if_exists='replace', index=False)
    ticket_data.to_sql('support_tickets', conn, if_exists='replace', index=False)
    
    # Save to CSV files as well
    customer_data.to_csv('data/customers.csv', index=False)
    usage_data.to_csv('data/usage_data.csv', index=False)
    ticket_data.to_csv('data/support_tickets.csv', index=False)
    
    conn.close()
    
    print(f"Database created successfully!")
    print(f"Customer records: {len(customer_data)}")
    print(f"Usage records: {len(usage_data)}")
    print(f"Support tickets: {len(ticket_data)}")
    print(f"Churn rate: {customer_data['churned'].mean():.2%}")
    
    return customer_data, usage_data, ticket_data

if __name__ == "__main__":
    create_database()
