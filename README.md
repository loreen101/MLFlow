# Bank Customer Churn Prediction

## Project Overview

This project aims to **predict bank customer churn** - identifying which customers are likely to leave the bank. Churn prediction is a critical business problem that helps financial institutions:

- Identify at-risk customers proactively
- Implement retention strategies before customers leave
- Improve customer lifetime value
- Reduce customer acquisition costs

The project uses machine learning to classify customers as likely to churn (Exited=1) or remain (Exited=0) based on their demographics and account characteristics.

## Dataset

The model is trained on **bank customer data** (`Churn_Modelling.csv`) containing the following features:

### Features Used:
- **CreditScore**: Customer's credit score
- **Geography**: Country (categorical: Germany, France, Spain)
- **Gender**: Customer gender (Male/Female)
- **Age**: Customer age in years
- **Tenure**: Number of years as a bank customer
- **Balance**: Account balance
- **NumOfProducts**: Number of products held with the bank
- **HasCrCard**: Whether customer has a credit card (binary)
- **IsActiveMember**: Whether customer is an active member (binary)
- **EstimatedSalary**: Customer's estimated yearly salary

### Target Variable:
- **Exited**: Whether customer churned (1) or remained (0)

## Data Preprocessing

### Data Balancing
The raw data exhibits class imbalance. To address this, the preprocessing pipeline:
- **Downsamples the majority class** to match the minority class
- Ensures balanced representation during model training
- Prevents bias toward the majority class

### Feature Engineering & Transformation

The preprocessing applies:

1. **Numerical Features** - StandardScaler normalization:
   - CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

2. **Categorical Features** - One-Hot Encoding:
   - Geography and Gender are encoded with drop='first' to avoid multicollinearity

3. **Train-Test Split**:
   - 70% training data
   - 30% test data
   - Random state: 1912 for reproducibility

The column transformer is serialized and logged as an artifact for production deployment.

## Models Trained

This project evaluated **3 machine learning models** for churn prediction:

### 1. **Logistic Regression**
- Simple linear classifier
- Good baseline model
- Interpretable coefficients
- Performance: Baseline performance

### 2. **Support Vector Machine (SVM) - RBF Kernel** ✅ **BEST PERFORMER**
- Kernel: RBF (Radial Basis Function)
- Gamma: 0.5
- Capable of handling non-linear decision boundaries
- **Best overall performance** on test set
- Excellent at separating churn and non-churn classes

### 3. **Random Forest Classifier** 🚀 **HIGH POTENTIAL**
- 5 estimators
- Max depth: 3
- Ensemble method combining multiple decision trees
- **Could potentially achieve better performance** with hyperparameter tuning:
  - Increasing number of estimators
  - Adjusting max_depth and min_samples_split
  - Feature importance analysis for better feature selection

## Model Performance Metrics

All models are evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives over predicted positives (minimize false alarms)
- **Recall**: True positives over actual positives (catch at-risk customers)
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)
- **Confusion Matrix**: Visualizes true positives, true negatives, false positives, false negatives

### Performance Summary:
| Model | Status |
|-------|--------|
| Logistic Regression | Baseline |
| SVM (RBF) | ⭐ Best Performance |
| Random Forest | 🚀 High Potential |

## Experiment Tracking with MLflow

This project uses **MLflow** for comprehensive experiment tracking:

### Logged Artifacts:
- **Models**: Trained model serialization with input/output schema inference
- **Column Transformer**: Preprocessing pipeline for feature transformation
- **Confusion Matrix**: PNG visualization of model performance
- **Training Dataset**: Input features and target variable logged

### Logged Parameters:
- Model hyperparameters (e.g., max_iter, kernel type, n_estimators)

### Logged Metrics:
- Accuracy, Precision, Recall, F1 Score

### Tagged Runs:
- Model type identification for easy filtering

**MLflow UI**: Available at `http://127.0.0.1:5000` for interactive experiment comparison

## Project Structure

```
MLFlow/
├── src/
│   └── train.py              # Main training script
├── dataset/
│   └── Churn_Modelling.csv   # Training dataset
├── X_train.csv               # Preprocessed training features
├── y_train.csv               # Preprocessed training target
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## How to Run

### Prerequisites
1. Activate the Python virtual environment:
   ```powershell
   ./Scripts/Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training a Model

1. Start the MLflow server:
   ```bash
   mlflow ui
   ```
   This launches the MLflow dashboard at `http://127.0.0.1:5000`

2. Run the training script:
   ```bash
   python ./src/train.py
   ```

3. View results in the MLflow UI to compare different model runs

## Key Findings & Recommendations

### Current Results:
✅ **SVM with RBF kernel** achieves the best performance with excellent class separation

### Future Improvements:
1. **Random Forest Optimization**:
   - Perform hyperparameter grid search to improve performance
   - Analyze feature importance scores
   - Consider increasing n_estimators and adjusting max_depth

2. **Additional Models to Consider**:
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks for complex pattern learning
   - Ensemble methods combining best models

3. **Data Enhancements**:
   - Feature engineering: interaction terms, polynomial features
   - Address remaining class imbalance with SMOTE
   - Explore feature scaling alternatives

4. **Production Deployment**:
   - Monitor model performance in production
   - Set up automated retraining pipelines
   - Implement A/B testing for model updates

## Dependencies

See `requirements.txt` for complete list. Key libraries:
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms and preprocessing
- **mlflow**: Experiment tracking and model management
- **matplotlib**: Visualization
- **numpy, scipy**: Numerical computing

## Author

Created as part of ML pipeline development and experiment tracking practice.

---

*Last Updated: February 2026*
