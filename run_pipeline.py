from pipelines.training_pipeline import train_pipelines
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipelines(data_path="./data/olist_customers_dataset.csv")