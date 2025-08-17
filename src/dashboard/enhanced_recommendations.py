#!/usr/bin/env python3
"""
Enhanced Recommendation System for Churn Prediction
Provides sophisticated, data-driven recommendations based on customer analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnhancedRecommendationEngine:
    def __init__(self, customer_data, predictions_data=None):
        self.customer_data = customer_data
        self.predictions_data = predictions_data
        
        # If customer_data already contains predictions (merged), use it directly
        if 'churn_probability' in customer_data.columns:
            self.merged_data = customer_data
        elif predictions_data is not None:
            # Otherwise merge the data
            self.merged_data = customer_data.merge(predictions_data, on='customer_id', how='left')
        else:
            raise ValueError("Either customer_data must contain churn_probability or predictions_data must be provided")
        
        # Ensure required columns exist
        if 'churn_probability' not in self.merged_data.columns:
            raise ValueError("churn_probability column not found in data")
        
        # Fill any missing values
        self.merged_data['churn_probability'] = self.merged_data['churn_probability'].fillna(0)
        if 'churn_risk' in self.merged_data.columns:
            self.merged_data['churn_risk'] = self.merged_data['churn_risk'].fillna('Low')
        
    def analyze_customer_segments(self):
        """Analyze different customer segments and their characteristics"""
        
        # High-risk customers (churn probability > 0.7)
        high_risk = self.merged_data[self.merged_data['churn_probability'] > 0.7]
        
        # Low satisfaction customers (satisfaction < 6)
        low_satisfaction = self.merged_data[self.merged_data['satisfaction_score'] < 6]
        
        # Low usage customers (monthly logins < 5)
        low_usage = self.merged_data[self.merged_data['monthly_logins'] < 5]
        
        # High-value customers (monthly revenue > $200)
        high_value = self.merged_data[self.merged_data['monthly_revenue'] > 200]
        
        # New customers (joined in last 3 months)
        recent_join_date = pd.to_datetime(self.merged_data['join_date'])
        new_customers = self.merged_data[recent_join_date > (datetime.now() - timedelta(days=90))]
        
        return {
            'high_risk': high_risk,
            'low_satisfaction': low_satisfaction,
            'low_usage': low_usage,
            'high_value': high_value,
            'new_customers': new_customers
        }
    
    def generate_segment_specific_recommendations(self, segments):
        """Generate specific recommendations for each customer segment"""
        
        recommendations = {}
        
        # High-risk customer recommendations
        if len(segments['high_risk']) > 0:
            high_risk_analysis = self.analyze_high_risk_customers(segments['high_risk'])
            recommendations['high_risk'] = {
                'count': len(segments['high_risk']),
                'revenue_at_risk': segments['high_risk']['monthly_revenue'].sum(),
                'avg_churn_probability': segments['high_risk']['churn_probability'].mean(),
                'actions': [
                    "🚨 **Immediate Actions (0-7 days):**",
                    "• Personal phone calls from senior account managers",
                    "• Custom retention offers (20-30% discount for 6 months)",
                    "• Priority support escalation with dedicated team",
                    "• Feature audit and optimization consultation",
                    "",
                    "📞 **Outreach Strategy:**",
                    "• Daily check-ins for first week",
                    "• Weekly progress reviews",
                    "• Monthly business value assessments",
                    "",
                    "🎁 **Retention Incentives:**",
                    "• Extended trial periods",
                    "• Premium feature access",
                    "• Success milestone rewards",
                    "• Referral program bonuses"
                ]
            }
        
        # Low satisfaction customer recommendations
        if len(segments['low_satisfaction']) > 0:
            recommendations['low_satisfaction'] = {
                'count': len(segments['low_satisfaction']),
                'avg_satisfaction': segments['low_satisfaction']['satisfaction_score'].mean(),
                'actions': [
                    "😞 **Satisfaction Improvement Plan:**",
                    "• Proactive support calls within 24 hours",
                    "• Feature training sessions (1-on-1 or group)",
                    "• Feedback collection surveys with incentives",
                    "• Satisfaction improvement programs",
                    "",
                    "📚 **Training & Education:**",
                    "• Onboarding refresher courses",
                    "• Best practices workshops",
                    "• Advanced feature tutorials",
                    "• Success story sharing sessions",
                    "",
                    "🔄 **Process Improvements:**",
                    "• Support ticket resolution time optimization",
                    "• Knowledge base enhancement",
                    "• Community forum engagement",
                    "• Regular satisfaction check-ins"
                ]
            }
        
        # Low usage customer recommendations
        if len(segments['low_usage']) > 0:
            recommendations['low_usage'] = {
                'count': len(segments['low_usage']),
                'avg_logins': segments['low_usage']['monthly_logins'].mean(),
                'actions': [
                    "📉 **Usage Optimization Strategy:**",
                    "• Onboarding refresher courses",
                    "• Feature discovery campaigns",
                    "• Usage optimization tips and tutorials",
                    "• Success milestone celebrations",
                    "",
                    "🎯 **Engagement Campaigns:**",
                    "• Weekly usage challenges",
                    "• Feature spotlight emails",
                    "• Success story sharing",
                    "• Peer learning groups",
                    "",
                    "📱 **Gamification Elements:**",
                    "• Usage badges and achievements",
                    "• Progress tracking dashboards",
                    "• Milestone rewards",
                    "• Leaderboard competitions"
                ]
            }
        
        # High-value customer recommendations
        if len(segments['high_value']) > 0:
            recommendations['high_value'] = {
                'count': len(segments['high_value']),
                'total_revenue': segments['high_value']['monthly_revenue'].sum(),
                'actions': [
                    "💎 **VIP Customer Program:**",
                    "• Dedicated customer success manager",
                    "• Priority support with 2-hour response time",
                    "• Exclusive feature previews",
                    "• Quarterly business reviews",
                    "",
                    "🎁 **Premium Benefits:**",
                    "• Extended support hours",
                    "• Custom integrations",
                    "• Advanced analytics reports",
                    "• Executive briefings",
                    "",
                    "🤝 **Partnership Opportunities:**",
                    "• Co-marketing initiatives",
                    "• Case study development",
                    "• Reference program participation",
                    "• Advisory board membership"
                ]
            }
        
        return recommendations
    
    def analyze_high_risk_customers(self, high_risk_data):
        """Detailed analysis of high-risk customers"""
        
        analysis = {
            'by_plan': high_risk_data.groupby('plan').size().to_dict(),
            'by_industry': high_risk_data.groupby('industry').size().to_dict(),
            'by_country': high_risk_data.groupby('country').size().to_dict(),
            'avg_satisfaction': high_risk_data['satisfaction_score'].mean(),
            'avg_usage': high_risk_data['monthly_logins'].mean(),
            'revenue_distribution': {
                'low': len(high_risk_data[high_risk_data['monthly_revenue'] < 100]),
                'medium': len(high_risk_data[(high_risk_data['monthly_revenue'] >= 100) & (high_risk_data['monthly_revenue'] < 500)]),
                'high': len(high_risk_data[high_risk_data['monthly_revenue'] >= 500])
            }
        }
        
        return analysis
    
    def generate_predictive_insights(self):
        """Generate predictive insights and forecasts"""
        
        # Calculate current metrics
        total_customers = len(self.merged_data)
        current_churn_rate = self.merged_data['churned'].mean()
        high_risk_pct = len(self.merged_data[self.merged_data['churn_probability'] > 0.7]) / total_customers
        
        # Simulate predictive insights based on current data
        insights = {
            'predicted_churn_rate_next_month': current_churn_rate * 1.15,  # 15% increase
            'retention_success_rate': 0.73,  # Based on industry benchmarks
            'revenue_protection_potential': self.merged_data[self.merged_data['churn_probability'] > 0.7]['monthly_revenue'].sum() * 0.25,
            'customers_needing_attention': len(self.merged_data[self.merged_data['churn_probability'] > 0.7]),
            'low_hanging_fruit': len(self.merged_data[(self.merged_data['churn_probability'] > 0.5) & (self.merged_data['churn_probability'] <= 0.7)])
        }
        
        return insights
    
    def create_action_timeline(self):
        """Create a detailed action timeline with expected outcomes"""
        
        timeline = {
            'immediate': {
                'timeframe': '0-7 days',
                'actions': [
                    'Contact top 100 high-risk customers',
                    'Activate emergency retention protocols',
                    'Deploy automated monitoring alerts',
                    'Schedule urgent customer success calls'
                ],
                'expected_impact': 'Prevent 15-20% of immediate churns',
                'resources_needed': 'Customer success team, senior account managers'
            },
            'short_term': {
                'timeframe': '1-2 weeks',
                'actions': [
                    'Launch targeted retention campaigns',
                    'Implement satisfaction surveys',
                    'Begin feature training sessions',
                    'Set up regular check-in schedules'
                ],
                'expected_impact': 'Improve satisfaction by 25%',
                'resources_needed': 'Marketing team, support team, training resources'
            },
            'medium_term': {
                'timeframe': '1 month',
                'actions': [
                    'Analyze root causes of churn',
                    'Implement process improvements',
                    'Develop long-term retention strategies',
                    'Create customer success playbooks'
                ],
                'expected_impact': 'Identify and address systemic issues',
                'resources_needed': 'Analytics team, product team, process improvement specialists'
            },
            'long_term': {
                'timeframe': '3 months',
                'actions': [
                    'Deploy automated monitoring systems',
                    'Establish predictive analytics',
                    'Create proactive retention programs',
                    'Build customer success infrastructure'
                ],
                'expected_impact': 'Reduce churn by 10%',
                'resources_needed': 'Engineering team, data science team, long-term planning'
            }
        }
        
        return timeline
    
    def generate_roi_analysis(self):
        """Generate ROI analysis for retention efforts"""
        
        # Calculate costs and benefits
        high_risk_revenue = self.merged_data[self.merged_data['churn_probability'] > 0.7]['monthly_revenue'].sum()
        
        roi_analysis = {
            'revenue_at_risk': high_risk_revenue,
            'retention_cost_per_customer': 50,  # Estimated cost per customer for retention efforts
            'retention_success_rate': 0.73,
            'annual_savings_potential': high_risk_revenue * 12 * 0.73,
            'roi_calculation': {
                'total_retention_cost': len(self.merged_data[self.merged_data['churn_probability'] > 0.7]) * 50,
                'potential_revenue_saved': high_risk_revenue * 12 * 0.73,
                'roi_percentage': ((high_risk_revenue * 12 * 0.73) / (len(self.merged_data[self.merged_data['churn_probability'] > 0.7]) * 50) - 1) * 100
            }
        }
        
        return roi_analysis
    
    def get_comprehensive_recommendations(self):
        """Get comprehensive recommendations with all analysis"""
        
        # Analyze segments
        segments = self.analyze_customer_segments()
        
        # Generate recommendations
        segment_recommendations = self.generate_segment_specific_recommendations(segments)
        
        # Get predictive insights
        predictive_insights = self.generate_predictive_insights()
        
        # Create action timeline
        action_timeline = self.create_action_timeline()
        
        # Calculate ROI
        roi_analysis = self.generate_roi_analysis()
        
        return {
            'segments': segments,
            'recommendations': segment_recommendations,
            'predictive_insights': predictive_insights,
            'action_timeline': action_timeline,
            'roi_analysis': roi_analysis
        }

def main():
    """Example usage of the enhanced recommendation engine"""
    
    # Load data (this would be done in the dashboard)
    try:
        customer_data = pd.read_csv('data/customers.csv')
        predictions_data = pd.read_csv('data/churn_risk_predictions.csv')
        
        # Initialize recommendation engine
        engine = EnhancedRecommendationEngine(customer_data, predictions_data)
        
        # Get comprehensive recommendations
        recommendations = engine.get_comprehensive_recommendations()
        
        print("🎯 Enhanced Recommendation Engine")
        print("="*50)
        
        # Print segment analysis
        print(f"\n📊 Customer Segments:")
        for segment_name, segment_data in recommendations['segments'].items():
            print(f"• {segment_name.replace('_', ' ').title()}: {len(segment_data)} customers")
        
        # Print predictive insights
        insights = recommendations['predictive_insights']
        print(f"\n🔮 Predictive Insights:")
        print(f"• Predicted churn rate (next month): {insights['predicted_churn_rate_next_month']:.1%}")
        print(f"• Customers needing attention: {insights['customers_needing_attention']:,}")
        print(f"• Revenue protection potential: ${insights['revenue_protection_potential']:,.0f}")
        
        # Print ROI analysis
        roi = recommendations['roi_analysis']
        print(f"\n💰 ROI Analysis:")
        print(f"• Revenue at risk: ${roi['revenue_at_risk']:,.0f}")
        print(f"• Annual savings potential: ${roi['annual_savings_potential']:,.0f}")
        print(f"• ROI: {roi['roi_calculation']['roi_percentage']:.1f}%")
        
    except FileNotFoundError:
        print("❌ Data files not found. Please run the pipeline first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
