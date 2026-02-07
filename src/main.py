from data_processing import *
from models import *

import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


os.environ["LOGNAME"] = "Loreen"


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    ### Set the experiment name
    mlflow.set_experiment("Churn Prediction Experiment")


    ### Start a new run and leave all the main function code as part of the experiment
    mlflow.start_run(run_name="CP_Run_3-Random_Forest")


    df = pd.read_csv("dataset/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    ### Log the max_iter parameter
    mlflow.log_param("max_iter", 1000)

    model = train(X_train, y_train, 'random_forest')

    y_pred = model.predict(X_test)

    ### Log metrics after calculating them
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)   
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")        
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    ### Log tag
    mlflow.set_tag("model_type", "svm")


    
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    
    # Log the image as an artifact in MLflow
    plt.savefig("./confusion_matrix_Random_Forest.png")
    mlflow.log_artifact("./confusion_matrix_Random_Forest.png", artifact_path="artifacts/confusion_matrix_Random_Forest")
    
    plt.show()
    mlflow.end_run()


if __name__ == "__main__":
    main()
