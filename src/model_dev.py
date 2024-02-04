import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

# *args collects data/variables & stores it in form of tuple
# **kwargs collects data & stores it in key-value pair(dictionary)

# Use *args for a bunch of items without individual names.
# Use **kwargs for items with specific labels.

class Model(ABC):
    """
    Abstract Class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the Model
        Args:
            X_train: Training data
            y_train: Training Labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Model training Completed!")
            return reg
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e