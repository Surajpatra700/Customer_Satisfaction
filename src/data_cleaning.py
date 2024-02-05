import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategies for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(["order_purchase_timestamp","order_approved_at", "order_delivered_carrier_date","order_delivered_customer_date", "order_estimated_delivery_date"], axis=1, errors='ignore')
            
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            # cols_to_drop = [
            #     "order_purchase_timestamp",
            #     "order_approved_at",
            #     "order_delivered_carrier_date",
            #     "order_delivered_customer_date",
            #     "order_estimated_delivery_date",
            # ]
            # data = data.drop(cols_to_drop, axis=1, errors='ignore')

            # logging.info("Imputing missing values...")
            # numerical_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
            # data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
            # data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include= [np.number]) # Only includes numerical for training to keep training simple
            cols_to_drop = ["customer_zip_code_prefix","order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy(DataStrategy):

    """
    Strategy for dividing data into train and Test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train & test
        """

        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        
# Utilising both of the startegies(DataPreProcessStrategy, DataDivideStrategy) into a single class
        
class DataCleaning:
    """
    Class for Cleaning data which processes the data and divides it into train& test
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy # Either of one startegy would be chosen at ones either DataPreProcessStrategy / DataDivideStrategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        # """
        # Handle data
        # """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(e)
            raise e
        
# if __name__ == "__main__":
#     data = pd.read_csv("C:\\Users\\suraj\\OneDrive\\Desktop\\Program Files\\ML_PIPELINE_MLOPS\\venv\\data\\olist_customers_dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data()
