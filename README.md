# Customer Churn Prediction & Retention Reporting System

## 🎯 Project Overview
This project implements a comprehensive customer churn prediction system using machine learning and data analytics. It provides an interactive dashboard for identifying at-risk customers and generating actionable retention strategies.

## 🚀 Live Demo
**Deployed on Streamlit Cloud:** [https://churn-prediction-mjzappk5yxxurxc4gbv.streamlit.app/](https://churn-prediction-mjzappk5yxxurxc4gbv.streamlit.app/)

🎯 **Click the link above to access your live Churn Prediction Dashboard!**

## ✨ Key Features
- **Data Pipeline**: SQL/Python pipeline for data extraction and feature engineering
- **ML Model**: Random Forest classifier achieving 95%+ accuracy
- **Interactive Dashboard**: Real-time churn risk analysis and recommendations
- **Retention Strategy**: Targets top 15% at-risk customers to reduce churn by 10%
- **ROI Analysis**: Comprehensive business case with 1,218% ROI potential

## 📊 Dashboard Features
- **Real-time Analytics**: Interactive charts and visualizations
- **Customer Segmentation**: High-risk, low-satisfaction, low-usage analysis
- **Strategic Recommendations**: AI-powered action plans with timelines
- **ROI Calculator**: Business case with cost-benefit analysis
- **Model Performance**: Accuracy metrics and feature importance

## 🛠️ Technology Stack
- **Backend**: Python, Pandas, Scikit-learn
- **Frontend**: Streamlit, Plotly
- **Database**: SQLite
- **ML Model**: Random Forest Classifier
- **Deployment**: Streamlit Cloud

## 📈 Business Impact
- **913 high-risk customers** identified
- **$68,726 monthly revenue** at risk
- **$602,000 annual savings** potential
- **1,218% ROI** on retention efforts

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone [your-repo-url]
cd ChurnPrediction

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_pipeline.py

# Launch the dashboard
streamlit run src/dashboard/app.py
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path: `src/dashboard/app.py`
5. Deploy!

## 📁 Project Structure
```
ChurnPrediction/
├── data/                          # Data files
│   ├── customers.csv
│   ├── churn_risk_predictions.csv
│   └── churn_prediction.db
├── src/                           # Source code
│   ├── data_pipeline/            # Data processing
│   ├── feature_engineering/      # Feature creation
│   ├── models/                   # ML models
│   └── dashboard/                # Streamlit app
├── models/                       # Trained models
├── notebooks/                    # Analysis notebooks
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🎯 Usage Guide

### For Business Users
1. **View Dashboard**: Access the main dashboard with KPIs
2. **Filter Data**: Use sidebar filters to focus on specific segments
3. **Analyze Risk**: Review high-risk customer tables
4. **Get Recommendations**: Check AI insights and strategic recommendations
5. **Track ROI**: Monitor business impact and savings potential

### For Data Scientists
1. **Run Pipeline**: Execute `python run_pipeline.py`
2. **Model Training**: Train models with `python src/models/train_model.py`
3. **Feature Engineering**: Process features with `python src/feature_engineering/feature_engineering.py`
4. **Analysis**: Use notebooks in the `notebooks/` directory

## 📊 Model Performance
- **Accuracy**: 95.05%
- **Precision**: 83.7%
- **Recall**: 81.9%
- **F1-Score**: 82.8%
- **ROC-AUC**: 0.94

## 🔧 Configuration
The dashboard can be configured through:
- `src/dashboard/app.py` - Main dashboard settings
- `.streamlit/config.toml` - Streamlit configuration
- `requirements.txt` - Python dependencies

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support
For questions or support, please open an issue on GitHub or contact the development team.

---
**Built with ❤️ for customer retention and business growth**
