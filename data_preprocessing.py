import os
import logging
import nltk
import nltk.downloader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
nltk.download("stopwords")
nltk.download("punkt")

# insure the log directory exist
log_dir= "logs"
os.makedirs(log_dir, exist_ok=True)

#setting up logger

logger= logging.getLogger("Data Prepocessing")
logger.setLevel(logging.DEBUG)

console_handler= logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path= os.path.join(log_dir, "data_preprocessing.log")
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter= logging.Formatter('%(asctime)s- %(name)s -%(levelname)s -%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """transform the input text into lower case , tokenization, removing stopwords, punctuation and stemming"""

    ps= PorterStemmer()
    text= text.lower()
    text= nltk.word_tokenize(text)
    #remove non alphanumeric tokens
    text= [word for word in text if word.isalnum()]
    #remove stopwords and punctuation
    text= [word for word in text if word not in stopwords.words("english") and word not in string.punctuation]
    #stem the words
    text=[ps.stem(word) for word in text]
    #join the tokens back into the single string
    return " ".join(text)

def data_preprocess(df, text_column="Message", target_column= "Category"):
    
    """Pre-processing the data frame by encoding the target column, removing duplicates and trasnforming the text column"""
    try:
        logger.debug("Starting preprocessing the data frame")
        #encode the target column
        encoder=LabelEncoder()
        df["Category"]= encoder.fit_transform(df["Category"])
        logger.debug("Target column encoded")

        #Remove duplicate rows
        df =df.drop_duplicates(keep="first")
        logger.debug("Duplicates removed")

        #Apply text transformation to the specified text column
        df.loc[:, "Message"]= df["Message"].apply(transform_text)
        logger.debug("Text column transformed")
        return df
   
    except KeyError as e:
        logger.error("column not found %s ", e)
        raise
    except Exception as e:
        logger.error("Error during text normalization : %s", e)
        raise

def main(text_column= "Message", target_column="Category"):
    """ Main function to load  raw data , process it and save the processed data"""

    try:
        #Fetch the data from raw

        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")

        #Transform the data

        train_processed_data=data_preprocess(train_data, text_column, target_column)
        test_processed_data = data_preprocess(test_data, text_column, target_column)

        # store the data inside the data path

        data_path= os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug("Processed data saved  to %s", data_path)

    except FileNotFoundError as e:
        logger.error("File not found %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data %s", e)

    except Exception as e:
        logger.error("Failed to complete the data transformation process %s", e)
        print(f"Error {e}")
    
if __name__=="__main__":
    main()


       
    
