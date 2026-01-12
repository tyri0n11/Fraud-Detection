# 00_eda.ipynb - Exploratory Data Analysis

## 📊 Overview
This notebook performs comprehensive exploratory data analysis (EDA) on the fraud detection dataset to understand data characteristics, identify patterns, and guide feature engineering decisions.

## 🎯 Objectives
- Understand the dataset structure and data quality
- Analyze fraud transaction patterns and characteristics
- Explore temporal patterns in fraudulent vs legitimate transactions
- Identify feature distributions and correlations
- Guide preprocessing and feature engineering strategies

## 📋 Key Sections

### 1. Data Loading and Overview
- Load the raw transaction dataset (`Final Transaction.csv`)
- Examine dataset structure, schema, and basic statistics
- Identify data types and missing values
- Calculate fraud rate and class distribution

### 2. Data Quality Assessment
- **Missing Values Analysis**: Check for null values across all columns
- **Data Type Validation**: Ensure proper data types for analysis
- **Duplicate Detection**: Identify and handle duplicate transactions
- **Outlier Detection**: Statistical analysis of extreme values

### 3. Fraud Pattern Analysis
- **Class Imbalance**: Analyze the proportion of fraud vs legitimate transactions
- **Amount Distribution**: Compare transaction amounts between fraud and legitimate
- **Temporal Patterns**: Fraud trends over time, days of week, hours
- **Statistical Comparison**: T-tests and distribution comparisons

### 4. Visual Analysis
- **Distribution Plots**: Histograms and box plots for key features
- **Time Series Analysis**: Fraud rates over time periods
- **Correlation Heatmap**: Feature correlation matrix
- **Scatter Plots**: Relationships between features and fraud

### 5. Feature Insights
- **Transaction Amount**: Range analysis and outlier patterns
- **Time Features**: Day, hour, weekend vs weekday patterns
- **Derived Features**: Log transformations and time-based features

## 🔍 Key Findings
- **Fraud Rate**: Approximately X% of transactions are fraudulent (highly imbalanced)
- **Amount Patterns**: Fraudulent transactions show distinct amount distributions
- **Temporal Patterns**: Fraud rates vary by time of day and day of week
- **Feature Correlations**: Identify most predictive features for modeling

## 📈 Visualizations Generated
- Transaction amount distributions by fraud class
- Time-based fraud rate analysis
- Feature correlation heatmaps
- Box plots for outlier detection
- Temporal trend analysis charts

## 🔧 Technical Requirements
```python
# Main libraries used
- pandas, numpy: Data manipulation and analysis
- matplotlib, seaborn: Data visualization
- pyspark: Big data processing
- datetime: Time-based analysis
```

## 🚀 How to Run
1. Ensure the raw data file `Final Transaction.csv` is in the `../data/raw/` directory
2. Run all cells sequentially
3. Generated insights will guide the feature engineering process

## 📝 Output
- Data quality report
- Statistical summaries by fraud class
- Visualization plots saved as images
- Insights document for next phases

## ⭐ Key Takeaways
The EDA reveals crucial insights about:
- Dataset size and fraud prevalence
- Most discriminative features for fraud detection
- Data preprocessing requirements
- Feature engineering opportunities
- Model selection guidance based on data characteristics

---
**Next Step**: Use these insights in [01_feature_engineering.ipynb](01_feature_engineering.ipynb) for creating predictive features.