# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites
- ✅ All data files generated (`data/` directory)
- ✅ Trained model files (`models/` directory)
- ✅ Dashboard application (`src/dashboard/app.py`)
- ✅ Requirements file (`requirements.txt`)

## 🎯 Quick Deployment Steps

### Step 1: Prepare Your GitHub Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Churn Prediction Dashboard"
   ```

2. **Create GitHub Repository**:
   - Go to [GitHub.com](https://github.com)
   - Click "New repository"
   - Name it: `churn-prediction-dashboard`
   - Make it public (for free Streamlit Cloud)
   - Don't initialize with README (we already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/churn-prediction-dashboard.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository: `churn-prediction-dashboard`
   - Set main file path: `src/dashboard/app.py`
   - Click "Deploy!"

3. **Wait for Deployment**:
   - Streamlit will automatically install dependencies
   - Build and deploy your app
   - You'll get a public URL like: `https://your-app-name.streamlit.app`

## 🔧 Configuration Options

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Environment Variables (Optional)
You can set these in Streamlit Cloud settings:
- `STREAMLIT_SERVER_PORT`: 8501
- `STREAMLIT_SERVER_HEADLESS`: true

## 📊 What Gets Deployed

### ✅ Files Included:
- `src/dashboard/app.py` - Main dashboard application
- `src/dashboard/enhanced_recommendations.py` - AI recommendations engine
- `data/customers.csv` - Customer data (10,000 records)
- `data/churn_risk_predictions.csv` - ML predictions
- `models/random_forest_model.pkl` - Trained model
- `requirements.txt` - Python dependencies

### 🎯 Features Available:
- **Interactive Analytics** - Real-time charts and visualizations
- **High-Risk Customer Analysis** - Detailed tables with search/sort
- **AI-Powered Insights** - Comprehensive recommendations
- **Strategic Action Plans** - ROI analysis and timelines
- **Model Performance** - Accuracy metrics and comparisons

## 🚨 Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   - Ensure all dependencies are in `requirements.txt`
   - Check that file paths are correct

2. **Data loading errors**:
   - Verify all CSV files are in the `data/` directory
   - Check file permissions

3. **Model loading errors**:
   - Ensure model files are in the `models/` directory
   - Check file sizes (should be several MB)

4. **Memory issues**:
   - Streamlit Cloud has memory limits
   - Consider reducing data size if needed

### Debug Commands:
```bash
# Test locally first
streamlit run src/dashboard/app.py

# Check file sizes
ls -la data/
ls -la models/

# Verify requirements
pip install -r requirements.txt
```

## 📈 Performance Optimization

### For Better Performance:
1. **Reduce data size** if needed:
   - Sample data for demo purposes
   - Use smaller datasets for testing

2. **Optimize loading**:
   - Cache expensive operations
   - Use lazy loading for large files

3. **Monitor usage**:
   - Check Streamlit Cloud dashboard
   - Monitor app performance

## 🔄 Updating Your App

### To Update After Changes:
1. **Make your changes** locally
2. **Test locally**:
   ```bash
   streamlit run src/dashboard/app.py
   ```
3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update dashboard features"
   git push
   ```
4. **Streamlit Cloud** will automatically redeploy

## 📞 Support

### If You Need Help:
1. **Check Streamlit Cloud logs** in the dashboard
2. **Review error messages** carefully
3. **Test locally** first
4. **Check Streamlit documentation**: [docs.streamlit.io](https://docs.streamlit.io)

### Common Error Solutions:
- **Import errors**: Check `requirements.txt`
- **File not found**: Verify file paths
- **Memory errors**: Reduce data size
- **Timeout errors**: Optimize code performance

## 🎉 Success Checklist

Before deploying, ensure you have:
- ✅ [ ] All data files in `data/` directory
- ✅ [ ] Trained model in `models/` directory
- ✅ [ ] `requirements.txt` with all dependencies
- ✅ [ ] `src/dashboard/app.py` as main file
- ✅ [ ] Code pushed to GitHub
- ✅ [ ] Streamlit Cloud app created
- ✅ [ ] App deployed successfully
- ✅ [ ] Dashboard loads without errors
- ✅ [ ] All features working correctly

## 🚀 Your App URL

Your live dashboard is available at:
```
https://churn-prediction-az3hjkvywwd5kxeoj7sfor.streamlit.app/
```

Share this URL with stakeholders, add it to your resume, or use it for presentations!

---

**🎯 Ready to deploy? Follow the steps above and your churn prediction dashboard will be live on Streamlit Cloud!**
