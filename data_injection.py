import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

#Ensure that log directory exist
log_dir= "logs"
os.makedirs(log_dir, exist_ok= True)

#Logging configuration
logger= logging.getLogger("Data Ingestion")
logger.setLevel(logging.DEBUG)

console_handler= logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path= os.path.join(log_dir, "data_ingestion.log")
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter= logging.Formatter('%(asctime)s- %(name)s -%(levelname)s -%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str)-> pd.DataFrame:
    """Load data from csv file"""
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Falied to parse the csv file : %s ", e)

    except Exception as e :
        logger.error("Unexpected error occurred while loading the data : %s", e )

def Data_preprocessing(df: pd.DataFrame)->pd.DataFrame:
    """ Processing the data"""
    try:
        #df.drop(columns=["Unnammed:2", "Unnammed :3", "Unnammed :4"], inplace=True)
        df.rename(columns={"Category":"Category", "Message":"Message"}, inplace=True)
        logger.debug("Data Processing completed")
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe : %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s",e)
        raise

def save_data(train_data :pd.DataFrame, test_data: pd.DataFrame, data_path: str)->None:
    """save the train and test data set"""
    try:
        raw_data_path= os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug("train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occured while saving the data %s", e)
        raise
def main():
    try:
        test_size= 0.2
        #data_path= "https://raw.githubusercontent.com/AmitK0105/Dataset/refs/heads/main/spam.csv"
        data_path= "E:/git-tutorial/MLOps-OOP/MLOps-Complete-Pipeline/Experiment/src/spam.csv"
        df= load_data(data_url=data_path)
        final_df= Data_preprocessing(df)
        train_data, test_data= train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path="./data")

    except Exception as e:
        logger.error("Failed to complete the data ingestion step : %s", e)
        print(f"Error ",e)

if __name__== "__main__":
    main()



