import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
#Ensure the "logs" directory exists
log_dir= "logs"
os.makedirs(log_dir, exist_ok=True)

#Logging Configuration
logger= logging.getLogger("Feature Engineering")
logger.setLevel(logging.DEBUG)

console_handler= logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)


log_file_path= os.path.join(log_dir, "Feature_Engineering.log")
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter= logging.Formatter('%(asctime)s- %(name)s -%(levelname)s -%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    """Load data from csv file"""
    try:
        df= pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaN filled from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file : %s", e)
        raise

def apply_tfidf(train_data :pd.DataFrame, test_data: pd.DataFrame, max_feature: int)->tuple:
    """Apply TFIDF to the data"""
    try:
        vectorizer=TfidfVectorizer(max_features=max_feature)
        x_train= train_data["Message"].values
        y_train= train_data["Category"].values
        x_test= test_data["Message"].values
        y_test= test_data["Category"].values

        x_train_bow= vectorizer.fit_transform(x_train)
        x_test_bow= vectorizer.fit_transform(x_test)

        train_df= pd.DataFrame(x_train_bow.toarray())
        train_df["Category"]= y_train

        test_df= pd.DataFrame(x_test_bow.toarray())
        test_df["Category"]= y_test

        logger.debug("TF-IDF applied and data transformed")

        return train_df,test_df
    except Exception as e:
        logger.error("Error during bag of words transformation %s", e)
        raise 

def save_data(df: pd.DataFrame, file_path: str)->None:
    """save the dataframe to a csv file"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)

    except Exception as e:
        logger.error("Unexpected error while saving the data %s", e)
        raise

def main():
    try:
        max_feature=50
        train_data= load_data("E:/git-tutorial/MLOps-OOP/MLOps-Complete-Pipeline/Experiment/src/train_processed.csv")
        test_data= load_data("E:/git-tutorial/MLOps-OOP/MLOps-Complete-Pipeline/Experiment/src/test_processed.csv")

        train_df, test_df= apply_tfidf(train_data, test_data, max_feature)

        save_data(train_df, os.path.join("./data", "features", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "features", "test_tfidf.csv"))

    except Exception as e:
        logger.error("Falied to comlete the feature engineering process %s", e)
        print(f"Error :", e)

if __name__== "__main__":
    main()
