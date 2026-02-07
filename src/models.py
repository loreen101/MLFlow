import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def LR(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)


    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    y_pred = log_reg.predict(X_train)  
    signature = infer_signature(X_train, y_pred)

    # Log model
    
    mlflow.sklearn.log_model(log_reg, artifact_path="models/logistic_regression", signature=signature)

    ### Log the data
    data = pd.concat([X_train, y_train], axis=1)
    dataset = mlflow.data.from_pandas(data)
    mlflow.log_input(dataset)

    return log_reg


def svm(X_train, y_train):
    """
    Train a support vector machine model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        SVC: trained support vector machine model
    """

    svm = SVC(kernel='rbf', gamma=0.5, max_iter=1000)
    svm.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    y_pred = svm.predict(X_train)  
    signature = infer_signature(X_train, y_pred)

    # Log model
    
    mlflow.sklearn.log_model(svm, artifact_path="models/svm", signature=signature)

    ### Log the data
    data = pd.concat([X_train, y_train], axis=1)
    dataset = mlflow.data.from_pandas(data)
    mlflow.log_input(dataset)

    return svm


def random_forest(X_train, y_train):
    """
    Train a random forest model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        RandomForestClassifier: trained random forest model
    """

    random_forest_classifier = RandomForestClassifier(n_estimators=5, max_depth=3)
    random_forest_classifier.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    y_pred = random_forest_classifier.predict(X_train)  
    signature = infer_signature(X_train, y_pred)

    # Log model
    
    mlflow.sklearn.log_model(random_forest_classifier, artifact_path="models/random_forest", signature=signature)

    ### Log the data
    data = pd.concat([X_train, y_train], axis=1)
    dataset = mlflow.data.from_pandas(data)
    mlflow.log_input(dataset)

    return random_forest_classifier


def train(X_train, y_train, method: str = "random_forest"):
    """
    Small convenience wrapper to match `main.py` which calls `train()`.

    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels
        method (str): one of 'random_forest', 'svm', 'lr'

    Returns:
        Trained model instance
    """
    method = method.lower()
    if method == "random_forest":
        return random_forest(X_train, y_train)
    if method in ("svm", "s"):
        return svm(X_train, y_train)
    if method in ("lr", "logistic", "logistic_regression"):
        return LR(X_train, y_train)
    raise ValueError(f"Unknown training method: {method}")
