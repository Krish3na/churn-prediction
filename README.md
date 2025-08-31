# Customer Churn Prediction & Retention Analytics System

## 🎯 Project Overview

A comprehensive **Machine Learning-powered customer churn prediction system** that combines **SQL/Python data pipelines**, **Random Forest classification**, and **interactive analytics dashboards** to identify at-risk customers and drive retention strategies. This project demonstrates end-to-end data science workflow from data engineering to model deployment.

**Live Demo:** [🚀 Interactive Dashboard](https://churn-prediction-az3hjkvywwd5kxeoj7sfor.streamlit.app/)

## 🏗️ Architecture & Technical Stack

### **Data Engineering Pipeline**
- **ETL Process**: SQL/Python pipeline for data extraction, transformation, and loading
- **Database**: SQLite with optimized schema for customer, usage, and support data
- **Feature Engineering**: 15+ derived features including engagement scores, risk indicators, and behavioral patterns
- **Data Quality**: Automated data validation and cleaning processes

### **Machine Learning Engine**
- **Algorithm**: Random Forest Classifier with hyperparameter optimization
- **Performance**: 95.05% accuracy, 83.7% precision, 81.9% recall
- **Feature Selection**: Top 10 most important features identified through SHAP analysis
- **Model Persistence**: Serialized models for production deployment

### **Analytics Dashboard**
- **Frontend**: Streamlit with Plotly interactive visualizations
- **Real-time Analytics**: Live KPIs, customer segmentation, and risk scoring
- **AI-Powered Insights**: Automated recommendation engine with ROI analysis
- **Deployment**: Streamlit Cloud with CI/CD integration

## 📊 Key Features & Capabilities

### **Data Pipeline Features**
- ✅ **Automated Data Extraction**: SQL queries for customer, usage, and support data
- ✅ **Feature Engineering**: Time-based aggregations, engagement metrics, risk scores
- ✅ **Data Validation**: Quality checks and missing value handling
- ✅ **Scalable Architecture**: Modular design for easy expansion

### **ML Model Capabilities**
- ✅ **High Accuracy**: 95.05% classification accuracy on test set
- ✅ **Feature Importance**: Identified key churn indicators
- ✅ **Hyperparameter Tuning**: GridSearchCV optimization
- ✅ **Cross-validation**: 5-fold CV for robust evaluation
- ✅ **Model Interpretability**: SHAP analysis for explainable AI

### **Analytics Dashboard Features**
- ✅ **Real-time KPIs**: Churn rate, revenue at risk, customer segments
- ✅ **Interactive Visualizations**: 3D scatter plots, geographic heatmaps, trend analysis
- ✅ **Customer Segmentation**: High-risk, low-satisfaction, low-usage analysis
- ✅ **AI Recommendations**: Automated action plans with timelines and ROI
- ✅ **Advanced Filtering**: Multi-dimensional data exploration

## 🎯 Business Impact & Results

### **Quantified Outcomes**
- **913 High-Risk Customers** identified with 85%+ churn probability
- **$68,726 Monthly Revenue** at risk from potential churn
- **$602,000 Annual Savings** potential through targeted retention
- **1,218% ROI** on retention efforts based on cost-benefit analysis
- **10% Churn Reduction** target through top 15% customer targeting

### **Operational Benefits**
- **Automated Risk Scoring**: Real-time customer churn probability
- **Targeted Retention**: Focus on highest-value at-risk customers
- **Data-Driven Decisions**: Evidence-based retention strategies
- **Scalable Solution**: Handles 10,000+ customer records efficiently

## 🛠️ Technology Stack

### **Backend & Data Processing**
- **Python 3.9+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SQLite**: Relational database management
- **SQLAlchemy**: Database ORM and query building

### **Machine Learning**
- **Scikit-learn**: ML algorithms and model training
- **Random Forest**: Primary classification algorithm
- **GridSearchCV**: Hyperparameter optimization
- **SHAP**: Model interpretability and feature importance
- **Joblib**: Model serialization and persistence

### **Data Visualization & Dashboard**
- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations
- **Matplotlib/Seaborn**: Static plotting and styling
- **Custom CSS**: Professional UI/UX design

### **Deployment & DevOps**
- **Streamlit Cloud**: Production deployment platform
- **Git/GitHub**: Version control and collaboration
- **Requirements.txt**: Dependency management
- **Docker**: Containerization (optional)

## 🚀 Quick Start Guide

### **Prerequisites**
```bash
Python 3.9+
Git
pip/conda
```

### **Installation & Setup**
```bash
# Clone the repository
git clone https://github.com/Krish3na/churn-prediction.git
cd churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py

# Launch dashboard locally
streamlit run src/dashboard/app.py
```

### **Data Pipeline Execution**
```bash
# Generate sample data
python src/data_pipeline/generate_sample_data.py

# Run data pipeline
python src/data_pipeline/main.py

# Feature engineering
python src/feature_engineering/feature_engineering.py

# Train ML model
python src/models/train_model.py
```

## 📁 Project Structure

```
churn-prediction/
├── 📊 data/                          # Data files and database
│   ├── customers.csv                 # Customer demographic data
│   ├── usage_data.csv               # Usage patterns and metrics
│   ├── support_tickets.csv          # Support interaction data
│   ├── churn_risk_predictions.csv   # ML model predictions
│   └── churn_prediction.db          # SQLite database
├── 🔧 src/                          # Source code
│   ├── data_pipeline/              # ETL and data processing
│   │   ├── generate_sample_data.py # Synthetic data generation
│   │   └── main.py                 # Main pipeline orchestration
│   ├── feature_engineering/        # Feature creation and selection
│   │   └── feature_engineering.py  # Feature engineering pipeline
│   ├── models/                     # ML model training
│   │   └── train_model.py          # Model training and evaluation
│   └── dashboard/                  # Streamlit application
│       ├── app.py                  # Main dashboard application
│       └── enhanced_recommendations.py # AI recommendation engine
├── 📈 models/                      # Trained ML models
│   ├── random_forest_model.pkl     # Serialized Random Forest model
│   ├── scaler.pkl                  # Feature scaling parameters
│   └── feature_importance.json     # Feature importance rankings
├── 📓 notebooks/                   # Jupyter analysis notebooks
│   ├── 01_data_exploration_and_cleaning.py
│   ├── 02_feature_engineering_analysis.py
│   └── 03_model_training_analysis.py
├── 📚 docs/                        # Documentation
│   ├── SQL_PIPELINE_DOCUMENTATION.md
│   ├── POWER_BI_DASHBOARD_GUIDE.md
│   └── PROJECT_COMPLETE_SUMMARY.md
├── 📋 requirements.txt             # Python dependencies
└── 📖 README.md                    # Project documentation
```

## 📈 Model Performance Metrics

### **Classification Performance**
- **Accuracy**: 95.05%
- **Precision**: 83.7%
- **Recall**: 81.9%
- **F1-Score**: 82.8%
- **ROC-AUC**: 0.94

### **Feature Importance (Top 10)**
1. **Monthly Usage Hours** (0.187)
2. **Support Tickets (30d)** (0.156)
3. **Customer Satisfaction Score** (0.134)
4. **Days Since Last Login** (0.112)
5. **Plan Type** (0.098)
6. **Engagement Score** (0.087)
7. **Payment Method** (0.076)
8. **Tenure (Months)** (0.065)
9. **Industry** (0.054)
10. **Country** (0.031)

## 🎯 Usage Scenarios

### **For Data Engineers**
- Study the SQL/Python ETL pipeline architecture
- Understand data modeling and schema design
- Learn automated data quality and validation processes
- Explore scalable data processing patterns

### **For ML Data Analysts**
- Analyze feature engineering techniques and business logic
- Review model performance evaluation methodologies
- Study customer segmentation and behavioral analysis
- Understand data-driven business insights generation

### **For AI/ML Engineers**
- Examine end-to-end ML pipeline implementation
- Study hyperparameter optimization and model selection
- Learn model deployment and production considerations
- Understand model interpretability and explainable AI

### **For Business Stakeholders**
- Access real-time customer churn risk analytics
- Generate targeted retention strategies
- Monitor KPIs and business impact metrics
- Make data-driven retention decisions

## 🔧 Configuration & Customization

### **Dashboard Configuration**
- Modify `src/dashboard/app.py` for UI/UX changes
- Update `.streamlit/config.toml` for deployment settings
- Customize `src/dashboard/enhanced_recommendations.py` for business logic

### **Model Configuration**
- Adjust hyperparameters in `src/models/train_model.py`
- Modify feature engineering in `src/feature_engineering/feature_engineering.py`
- Update data pipeline in `src/data_pipeline/main.py`

### **Data Configuration**
- Modify data schema in `src/data_pipeline/generate_sample_data.py`
- Update database queries in `src/data_pipeline/main.py`
- Customize feature calculations for your business domain

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Guidelines**
- Follow PEP 8 Python style guidelines
- Add comprehensive docstrings and comments
- Include unit tests for new features
- Update documentation for any changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/Krish3na/churn-prediction/issues)
- **Documentation**: Check the `docs/` directory for detailed guides
- **Live Demo**: [Interactive Dashboard](https://churn-prediction-az3hjkvywwd5kxeoj7sfor.streamlit.app/)

---

## 🎯 **Perfect for Your Target Roles**

This project demonstrates:
- **Data Engineering**: ETL pipelines, database design, data quality
- **ML Data Analysis**: Feature engineering, statistical analysis, business insights
- **AI/ML Engineering**: Model development, deployment, production ML systems
- **GenAI Engineering**: Automated insights, recommendation systems, AI applications

**Ready to showcase your data science and ML engineering skills!** 🚀
