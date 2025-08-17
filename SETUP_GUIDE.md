# 🚀 Churn Prediction Project Setup Guide

## Quick Start

### Option 1: One-Command Setup (Recommended)
```bash
python run_pipeline.py
```

### Option 2: Step-by-Step Setup
Follow the detailed instructions below.

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB free space
- **OS**: Windows, macOS, or Linux

### Python Installation
If you don't have Python installed:
1. Download from [python.org](https://www.python.org/downloads/)
2. Ensure "Add Python to PATH" is checked during installation
3. Verify installation: `python --version`

---

## 🔧 Installation Steps

### Step 1: Clone/Download Project
```bash
# If using git
git clone <repository-url>
cd ChurnPrediction

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Sample Data
```bash
python src/data_pipeline/generate_sample_data.py
```

### Step 4: Run Data Pipeline
```bash
python src/data_pipeline/main.py
```

### Step 5: Feature Engineering
```bash
python src/feature_engineering/feature_engineering.py
```

### Step 6: Train Model
```bash
python src/models/train_model.py
```

### Step 7: Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 📊 What You'll Get

### Generated Files
- **Data Files**: Customer data, usage patterns, support tickets
- **Model Files**: Trained Random Forest model, feature importance
- **Predictions**: Churn risk scores for all customers
- **Visualizations**: Performance plots and analysis charts

### Dashboard Features
- **Interactive Dashboard**: Real-time churn risk monitoring
- **Customer Segments**: Plan, industry, geographic analysis
- **Risk Distribution**: Visual charts and statistics
- **High-Risk Customers**: Top 15% at-risk customer list
- **Retention Insights**: Actionable recommendations

---

## 🎯 Model Performance

### Expected Results
- **Accuracy**: ~85%
- **F1-Score**: ~83%
- **ROC-AUC**: ~0.89
- **Processing Time**: 2-5 minutes for full pipeline

### Key Features
- **10,000 customers** with realistic data
- **50+ engineered features**
- **Random Forest model** with hyperparameter tuning
- **Interactive visualizations**

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. Memory Issues
```bash
# Solution: Reduce data size in generate_sample_data.py
# Change n_customers=10000 to n_customers=5000
```

#### 3. Dashboard Not Loading
```bash
# Solution: Check if port 8501 is available
# Alternative: streamlit run src/dashboard/app.py --server.port 8502
```

#### 4. Model Training Slow
```bash
# Solution: Disable hyperparameter tuning
# Edit train_model.py: tune_hyperparameters=False
```

### Error Messages

#### "ModuleNotFoundError"
- Ensure you're in the project root directory
- Check Python environment and dependencies

#### "Permission Denied"
- Run as administrator (Windows)
- Use `sudo` (Linux/Mac)

#### "Port Already in Use"
- Kill existing Streamlit processes
- Use different port: `--server.port 8502`

---

## 📈 Understanding the Results

### Model Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Business Metrics
- **Churn Rate**: Percentage of customers who churned
- **High-Risk Customers**: Top 15% with highest churn probability
- **Revenue at Risk**: Monthly revenue from high-risk customers
- **Retention Opportunities**: Customers needing attention

### Dashboard Sections
1. **Risk Analysis**: Churn probability distribution
2. **Customer Segments**: Demographics and behavior
3. **Usage Patterns**: Login frequency and engagement
4. **High-Risk Customers**: Actionable customer list
5. **Insights**: Recommendations and strategies

---

## 🎓 Learning Resources

### Key Concepts
- **Churn Prediction**: Identifying customers likely to leave
- **Feature Engineering**: Creating predictive variables
- **Random Forest**: Ensemble machine learning algorithm
- **Business Intelligence**: Data-driven decision making

### Technical Skills Demonstrated
- **Python**: Data manipulation and analysis
- **Machine Learning**: Model training and evaluation
- **Data Visualization**: Interactive dashboards
- **SQL**: Database operations and queries

---

## 🚀 Next Steps

### For Interviews
1. **Understand the Code**: Review each module thoroughly
2. **Explain the Process**: Be ready to walk through the pipeline
3. **Discuss Business Impact**: Focus on ROI and outcomes
4. **Show Technical Depth**: Explain model choices and trade-offs

### For Portfolio
1. **Customize the Data**: Use your own datasets
2. **Add Features**: Implement additional algorithms
3. **Deploy Online**: Use Streamlit Cloud or Heroku
4. **Document Everything**: Create detailed README files

### For Production
1. **Real Data Integration**: Connect to actual databases
2. **API Development**: Create prediction endpoints
3. **Monitoring**: Add model performance tracking
4. **Automation**: Schedule regular retraining

---

## 📞 Support

### Getting Help
- Check the troubleshooting section above
- Review error messages carefully
- Ensure all dependencies are installed
- Verify file paths and permissions

### Common Questions

**Q: Can I use my own data?**
A: Yes! Replace the sample data generation with your data loading logic.

**Q: How do I improve the model?**
A: Try different algorithms, add more features, or tune hyperparameters.

**Q: Can I deploy this online?**
A: Yes! Streamlit apps can be deployed on Streamlit Cloud, Heroku, or AWS.

**Q: What if the model accuracy is low?**
A: Check data quality, feature engineering, and try different algorithms.

---

## 🎯 Success Checklist

- [ ] All dependencies installed successfully
- [ ] Sample data generated (10,000 customers)
- [ ] Data pipeline completed without errors
- [ ] Feature engineering created 50+ features
- [ ] Model trained with >80% accuracy
- [ ] Dashboard launches and displays data
- [ ] Can identify high-risk customers
- [ ] Understand the business impact

**Congratulations!** You now have a complete churn prediction system ready for interviews and portfolio demonstrations.
