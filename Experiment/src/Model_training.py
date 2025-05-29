import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

#Ensure that "logs" directory exist

log_dir= "logs"
os.makedirs(log_dir, exist_ok=True)

#Logging Configuration
logger= logging.getLogger("Model Development")
logger.setLevel(logging.DEBUG)

console_handler= logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)


log_file_path= os.path.join(log_dir, "Model_development.log")
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter= logging.Formatter('%(asctime)s- %(name)s -%(levelname)s -%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str)->pd.DataFrame:
    """ Load data from csv file """

    try:
        df= pd.read_csv(file_path)
        logger.debug("Data Loaded from %s with shape %s", file_path, df.shape)
        return df
    except pd.errors.ParserError as E:
        logger.error("Failed to parse the csv file %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data %s", e)
        raise
def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict)->RandomForestClassifier:
       

    try:
     if x_train.shape[0] !=y_train.shape[0]:
          raise ValueError("The number of samples in x_train & y_train must be the same")
     logger.debug("Initialize RandomForest Model with parameters %s", params)
     clf= RandomForestClassifier(n_estimators=params["n_estimators"], random_state=params["random_state"])
     logger.debug("Model training started with %d samples", x_train.shape[0])
     clf.fit(x_train, y_train)
     logger.debug("Model training completed")
     return clf

    except ValueError as e:
     logger.error("Value error during the model training %s",e)
     raise
    except Exception as e:
     logger.error("Error during model training  %s", e)
     raise

def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)

    except FileNotFoundError as e:
        logger.error("File path not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred while saving the model: %s", e)
        raise
   
   
def main():
    try:
        params = {"n_estimators": 25, "random_state": 2}
        
        train_data = load_data(r"E:\git-tutorial\MLOps-OOP\MLOps-Complete-Pipeline\Experiment\src\data\processed\train_tfidf.csv")
        
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values  # corrected to last column for labels
        
        clf = train_model(x_train, y_train, params)
        
        model_save_path = "models/model.pkl"
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print("Error:", e)


if __name__=="__main__":
   main()

      
     


