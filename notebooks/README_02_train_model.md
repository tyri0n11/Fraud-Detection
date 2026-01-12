# 02_train_model.ipynb - Model Training

## 🎯 Overview
This notebook trains baseline machine learning models for fraud detection using the engineered features from the previous step. It implements three different algorithms with proper time-based data splitting to prevent temporal leakage.

## 🔮 Objectives
- Train baseline models for fraud detection
- Implement time-based train/validation/test splits
- Compare multiple machine learning algorithms
- Establish performance benchmarks
- Save trained models for evaluation

## 📋 Key Sections

### 1. Environment Setup & Data Loading
- Initialize Spark Session with ML-optimized configuration  
- Load engineered features from parquet file
- Verify data quality and feature completeness

### 2. Time-based Data Splitting
```python
# Temporal split to prevent data leakage
Train:      < 2023-06-01  # Historical data for training
Validation: 2023-06-01 to 2023-07-01  # Recent data for tuning  
Test:       >= 2023-07-01  # Future data for final evaluation
```

**Why Time-based Splitting?**
- Prevents temporal leakage in fraud detection
- Simulates real-world deployment scenario
- Maintains chronological order of transactions
- More realistic performance estimates

### 3. Feature Preparation
```python
# Feature vector assembly
enhanced_feature_cols = [
    "TX_AMOUNT", "LOG_TX_AMOUNT", "TX_TIME_SECONDS", 
    "TX_TIME_DAYS", "TX_HOUR", "IS_NIGHT", "IS_WEEKEND"
]

# Vector assembly for Spark ML
VectorAssembler(inputCols=enhanced_feature_cols, outputCol="features")
```

### 4. Model Training

#### 4.1 Logistic Regression
```python
# Configuration
- Regularization: L2 (Ridge)
- Max iterations: 100
- Feature scaling: StandardScaler
- Solver: Limited-memory BFGS
```

#### 4.2 Random Forest
```python
# Configuration  
- Number of trees: 100
- Max depth: 5
- Min instances per node: 1
- Bootstrap sampling: True
```

#### 4.3 Gradient Boosted Trees
```python
# Configuration
- Max iterations: 100  
- Max depth: 4
- Step size: 0.1
- Loss function: Logistic
```

### 5. Model Pipelines
Each model uses a complete MLlib Pipeline:
```python
Pipeline(stages=[
    VectorAssembler(),     # Feature assembly
    StandardScaler(),      # Feature scaling (for LR only)
    Classifier()           # ML algorithm
])
```

## 📊 Data Distribution

### Dataset Splits
| Split | Time Period | Transactions | Fraud Rate |
|-------|-------------|--------------|------------|
| Train | < 2023-06-01 | ~XXX,XXX | X.XX% |
| Validation | 2023-06-01 - 2023-07-01 | ~XX,XXX | X.XX% |
| Test | >= 2023-07-01 | ~XX,XXX | X.XX% |

### Class Imbalance Handling
- **Strategy**: Cost-sensitive learning via class weights
- **Baseline**: Use natural class distribution
- **Monitoring**: Track precision/recall balance

## 🏆 Baseline Results

### Model Performance Summary
| Model | AUC | PR-AUC | Precision | Recall | F1-Score |
|-------|-----|--------|-----------|---------|----------|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Gradient Boosted Trees | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

## 🔧 Technical Implementation

### Spark ML Configuration
```python
# Optimized for ML workloads
.config("spark.sql.adaptive.enabled", "true")
.config("spark.sql.adaptive.coalescePartitions.enabled", "true")  
.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

### Evaluation Metrics
- **ROC-AUC**: Overall discrimination ability
- **PR-AUC**: Performance on imbalanced data
- **Precision**: False positive control
- **Recall**: Fraud detection rate  
- **F1-Score**: Balanced precision/recall

### Model Persistence
```python
# Save trained models
model.write().overwrite().save(f"../models/{model_name}_baseline_model")
```

## 🎛️ Hyperparameters

### Default Settings (Baseline)
- **Logistic Regression**: Default regularization, 100 iterations
- **Random Forest**: 100 trees, depth 5, default splits
- **GBT**: 100 iterations, depth 4, learning rate 0.1

### Rationale
- Start with reasonable defaults
- Focus on proper evaluation setup
- Optimize in next phase (hyperparameter tuning)

## 📈 Performance Insights

### Key Observations
- **Best Performer**: [Model name] with AUC = X.XXX
- **Precision Leader**: [Model name] for minimizing false positives
- **Recall Champion**: [Model name] for catching fraud cases
- **Speed**: Random Forest fastest training time
- **Stability**: GBT most consistent across splits

### Business Impact
- **True Positives**: Successfully detected fraud cases
- **False Positives**: Legitimate transactions flagged
- **Cost Analysis**: Impact on customer experience vs fraud loss

## 🔧 Technical Requirements
```python
# Dependencies
- pyspark.ml: Machine learning algorithms
- pyspark.ml.classification: Specific classifiers
- pyspark.ml.evaluation: Performance metrics
- pyspark.ml.feature: Feature transformations
```

## 🚀 How to Run
1. Ensure engineered features are available: `../data/features/fraud_features_v2.parquet`
2. Run cells in sequence for reproducible training
3. Monitor Spark UI for training progress and resource usage
4. Verify model files are saved to `../models/` directory

## 📝 Outputs

### Trained Models
- `../models/lr_baseline_model/`: Logistic Regression pipeline
- `../models/rf_baseline_model/`: Random Forest pipeline  
- `../models/gbt_baseline_model/`: Gradient Boosted Trees pipeline

### Performance Metrics
- Validation metrics for model comparison
- Confusion matrices for each model
- Training time and resource usage statistics

## ⚡ Performance Notes
- **Training Time**: ~X minutes per model
- **Memory Usage**: Peak memory consumption tracking
- **Scalability**: Designed for larger datasets
- **Reproducibility**: Fixed random seeds for consistent results

## ⭐ Key Achievements
✅ **Time-based splitting** prevents temporal leakage  
✅ **Multiple algorithms** for comparison baseline  
✅ **Proper evaluation** with relevant metrics  
✅ **Model persistence** for downstream usage  
✅ **Scalable training** on distributed platform  

---
**Next Step**: Detailed model evaluation and analysis in [03_model_evaluation.ipynb](03_model_evaluation.ipynb).