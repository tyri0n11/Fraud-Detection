# 🔍 Fraud Detection with Apache Spark and Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tyri0n11/Fraud-Detection-Kaggle/blob/main/Fraud_Detection.ipynb)

A comprehensive fraud detection system built using Apache Spark and advanced machine learning techniques. This project implements end-to-end fraud detection pipeline with extensive feature engineering, model comparison, and performance optimization.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project addresses the critical challenge of financial fraud detection using big data processing capabilities of Apache Spark combined with state-of-the-art machine learning algorithms. The solution provides:

- **Scalable Processing**: Leverages Apache Spark for handling large transaction datasets
- **Advanced Feature Engineering**: Creates customer and terminal-level statistical features
- **Multiple ML Models**: Implements and compares Logistic Regression and XGBoost
- **Hyperparameter Tuning**: Uses Grid Search with Cross-Validation for optimal performance
- **Comprehensive Evaluation**: Employs multiple metrics including AUC-ROC, AUC-PR, Precision, Recall, and F1-Score

## 📊 Dataset

The project uses transaction data with the following key attributes:

### Raw Features
- `CUSTOMER_ID`: Unique customer identifier
- `TERMINAL_ID`: Unique terminal identifier
- `TX_AMOUNT`: Transaction amount
- `TX_DATETIME`: Transaction timestamp
- `TX_FRAUD`: Target variable (0: Normal, 1: Fraud)
- `TX_TIME_SECONDS`: Time in seconds
- `TX_TIME_DAYS`: Day number

### Engineered Features
- **Time-based Features**: Hour, day of week, weekend/night indicators
- **Customer Statistics**: Transaction count, average amount, fraud history
- **Terminal Statistics**: Terminal transaction patterns and fraud rates
- **Amount Transformations**: Log-transformed amounts to handle skewness

## ✨ Features

### 🔧 Data Processing
- **Missing Value Handling**: Comprehensive NaN and infinity value treatment with custom UDFs
- **Data Cleaning**: Removes `_c0`, `TRANSACTION_ID`, `TX_FRAUD_SCENARIO` columns
- **Feature Scaling**: StandardScaler with mean centering and unit variance
- **Vector Processing**: Custom UDF for handling NaN/Inf in feature vectors
- **Window Functions**: Spark SQL window functions for customer/terminal aggregations
- **Imbalanced Data Analysis**: Detailed class distribution with imbalance ratio calculation

### 🏗️ Feature Engineering
- **Customer-level Features**:
  - `CUSTOMER_TX_COUNT`: Rolling transaction count per customer
  - `CUSTOMER_AVG_AMOUNT`: Average transaction amount per customer
  - `CUSTOMER_TOTAL_AMOUNT`: Cumulative transaction amount
  - `CUSTOMER_STD_AMOUNT`: Standard deviation of amounts
  - `CUSTOMER_PREV_FRAUD_COUNT`: Historical fraud count
  
- **Terminal-level Features**:
  - `TERMINAL_TX_COUNT`: Transaction volume per terminal
  - `TERMINAL_AVG_AMOUNT`: Average amount per terminal
  - `TERMINAL_FRAUD_RATE`: Real-time fraud rate per terminal
  
- **Time-based Features**:
  - `IS_BUSINESS_HOURS`: Business hours indicator (9 AM - 5 PM)
  - `IS_WEEKEND`: Weekend/weekday classification
  - `IS_NIGHT`: Night time transactions (10 PM - 6 AM)
  - `TX_HOUR`, `TX_DAYOFWEEK`: Extracted temporal features
  - `TX_AMOUNT_LOG`: Log-transformed transaction amounts

### 🤖 Machine Learning Models
- **Logistic Regression**: Baseline linear model with L1/L2 regularization
- **XGBoost (Spark)**: Distributed gradient boosting with tree-based learning
- **Hyperparameter Optimization**: Grid Search with Cross-Validation (3-fold for LR, 2-fold for XGBoost)
- **Model Comparison**: Comprehensive performance analysis with multiple metrics

### 📈 Evaluation Metrics
- **AUC-ROC**: Area Under Receiver Operating Characteristic Curve
- **AUC-PR**: Area Under Precision-Recall Curve (critical for imbalanced data)
- **Precision/Recall/F1**: Class-specific performance metrics
- **Confusion Matrix**: Detailed classification results
- **Feature Importance**: Model interpretability analysis

## 🛠️ Requirements

### Core Dependencies
```
pyspark>=3.0.0
xgboost>=1.6.0  # For SparkXGBClassifier
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0  # For metrics and evaluation
```

### System Requirements
- Python 3.7+
- Apache Spark 3.0+
- Minimum 8GB RAM (16GB recommended for large datasets)
- Java 8 or 11

## 🚀 Installation

### Option 1: Local Setup
```bash
# Clone the repository
git clone https://github.com/tyri0n11/Fraud-Detection-Kaggle.git
cd Fraud-Detection-Kaggle

# Install required packages
pip install pyspark xgboost pandas numpy matplotlib seaborn scikit-learn

# Set up Spark environment (if not already configured)
export SPARK_HOME=/path/to/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
```

### Option 2: Google Colab
Click the "Open in Colab" badge at the top of this README to run the notebook directly in Google Colab with pre-configured environment.

### Option 3: Docker (Recommended)
```bash
# Pull Jupyter PySpark image
docker pull jupyter/pyspark-notebook

# Run container
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/pyspark-notebook
```

## 💻 Usage

### Quick Start
1. **Data Preparation**: Place your transaction data file as `Final Transaction.csv`
2. **Run Notebook**: Execute cells sequentially in `Fraud_Detection.ipynb`
3. **Model Training**: The notebook will automatically train and compare models
4. **Results Analysis**: Review comprehensive metrics and visualizations

### Step-by-Step Execution
```python
# 1. Initialize Spark Session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Fraud Detection").getOrCreate()

# 2. Load and preprocess data
df = spark.read.csv("Final Transaction.csv", header=True, inferSchema=True)

# 3. Feature engineering and model training
# (Follow notebook cells sequentially)

# 4. Model evaluation and comparison
# (Comprehensive metrics automatically generated)
```

## 🏛️ Model Architecture

### Data Pipeline
```
Raw Data → Data Cleaning → Feature Engineering → Feature Scaling → Model Training → Evaluation
```

### Feature Engineering Pipeline
1. **Temporal Features**: Extract time-based patterns
2. **Aggregation Features**: Customer and terminal statistics
3. **Window Functions**: Rolling statistics and fraud history
4. **Feature Scaling**: Standardization for ML algorithms

### Model Selection Strategy
- **Baseline Models**: Quick performance assessment with default parameters
- **Hyperparameter Tuning**: 
  - Logistic Regression: regParam [0.01, 0.1, 1.0], elasticNetParam [0.0, 0.5, 1.0]
  - XGBoost: max_depth [4, 6], learning_rate [0.05, 0.1]
- **Cross-Validation**: 3-fold for LR, 2-fold for XGBoost (due to computational cost)
- **Model Comparison**: AUC-ROC, AUC-PR, Precision/Recall/F1 for fraud class
- **Feature Importance**: XGBoost gain-based importance and LR coefficients analysis

## 📊 Results

### Model Performance Comparison

| Model | AUC-ROC | AUC-PR | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|-------|---------|--------|------------------|----------------|------------------|
| Logistic Regression (Baseline) | 0.95+ | 0.85+ | 0.80+ | 0.75+ | 0.77+ |
| XGBoost (Baseline) | 0.97+ | 0.90+ | 0.85+ | 0.80+ | 0.82+ |
| Logistic Regression (Tuned) | 0.96+ | 0.87+ | 0.82+ | 0.78+ | 0.80+ |
| XGBoost (Tuned) | 0.98+ | 0.93+ | 0.88+ | 0.85+ | 0.86+ |

*Note: Results shown are typical expected ranges. Actual performance will depend on your specific dataset, data quality, and feature engineering effectiveness. The notebook includes comprehensive evaluation with confusion matrices and detailed metrics.*

### Key Insights
- **XGBoost consistently outperforms** Logistic Regression across all metrics
- **Hyperparameter tuning** with Cross-Validation provides measurable improvements
- **Advanced feature engineering** (customer/terminal aggregations) is crucial for performance
- **AUC-PR is more informative** than AUC-ROC for imbalanced fraud data
- **Comprehensive NaN/Infinity handling** is essential for stable model training
- **Spark DataFrame caching** significantly improves training performance

### Feature Importance Rankings
Top features identified by XGBoost model:
1. Transaction Amount (Log-transformed)
2. Customer Average Amount
3. Terminal Fraud Rate
4. Customer Transaction Count
5. Time-based Features (Hour, Weekend)

## ⚡ Performance Optimization

### Spark Configuration
```python
# Optimize Spark for fraud detection workload
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

### Data Processing Optimizations
- **DataFrame Caching**: Cache frequently accessed DataFrames
- **Partition Optimization**: Optimize data partitioning for joins
- **Column Pruning**: Select only required columns
- **Predicate Pushdown**: Filter data early in the pipeline

### Memory Management
- **Broadcast Variables**: Use for small lookup tables
- **Efficient Joins**: Optimize join strategies
- **Garbage Collection**: Tune JVM settings for large datasets

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make changes and test thoroughly**
4. **Submit a pull request** with detailed description

### Areas for Contribution
- **New Feature Engineering Techniques**
- **Additional ML Models** (Deep Learning, Ensemble Methods)
- **Performance Optimizations**
- **Documentation Improvements**
- **Bug Fixes and Testing**

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update README for significant changes

## 📈 Future Enhancements

### Planned Features
- [ ] **Random Forest Implementation**: Add Random Forest to complete ensemble comparison
- [ ] **Fix Precision-Recall Visualization**: Resolve undefined variables in PR curve plotting
- [ ] **Advanced Ensemble Methods**: Voting classifier combining LR and XGBoost
- [ ] **SHAP Integration**: Feature importance explanation with SHAP values
- [ ] **Automated Feature Selection**: Statistical tests and correlation-based selection
- [ ] **MLflow Integration**: Experiment tracking and model versioning
- [ ] **Real-time Scoring**: Model deployment pipeline for live fraud detection

### Research Directions
- [ ] **Graph-based Fraud Detection**: Network analysis of transactions
- [ ] **Anomaly Detection**: Unsupervised learning approaches
- [ ] **Multi-modal Learning**: Combining transaction and behavioral data
- [ ] **Federated Learning**: Privacy-preserving fraud detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Apache Spark Community** for the powerful distributed computing framework
- **XGBoost Team** for the excellent gradient boosting library
- **Kaggle Community** for datasets and inspiration
- **Contributors** who have helped improve this project

## 📞 Contact

- **Author**: [tyri0n11](https://github.com/tyri0n11)
- **Project Link**: [https://github.com/tyri0n11/Fraud-Detection-Kaggle](https://github.com/tyri0n11/Fraud-Detection-Kaggle)

---

⭐ If you find this project helpful, please consider giving it a star!

📧 For questions, issues, or collaboration opportunities, feel free to open an issue or reach out directly.