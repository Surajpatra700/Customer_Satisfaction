import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: " + str(mse))
            return mse
        except Exception as e:
            logging.error(e)
            raise e
        
class R2Score(Evaluation):   
    """
    Evaluation Strategy that uses r2 score Error
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error(e)
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating RMSE")
            # rmse = mean_squared_error(y_true, y_pred, squared=False)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error(e)
            raise e