# Predicting Customer Satisfaction Before Purchase

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

This project predicts a customer's satisfaction score for their next order based on historical data using the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The goal is to build a production-ready machine learning pipeline using [ZenML](https://zenml.io/) to predict customer satisfaction scores based on features like order status, price, payment, and more.

## 🚀 Key Features

- **End-to-End ML Pipeline**: From data ingestion to model deployment.
- **MLflow Integration**: For tracking experiments, metrics, and model deployment.
- **Streamlit Demo**: A live application to showcase predictions.
- **Continuous Deployment**: Automatically deploy models that meet performance thresholds.

## 🛠️ Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zenml-io/zenml-projects.git
   cd zenml-projects/customer-satisfaction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install ZenML Server** (Optional for Dashboard):
   ```bash
   pip install zenml["server"]
   zenml up
   ```

4. **Install MLflow Integration**:
   ```bash
   zenml integration install mlflow -y
   ```

5. **Configure ZenML Stack**:
   ```bash
   zenml experiment-tracker register mlflow_tracker --flavor=mlflow
   zenml model-deployer register mlflow --flavor=mlflow
   zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
   ```

## 🧠 Pipeline Overview

### Training Pipeline
1. **Ingest Data**: Load the dataset into a DataFrame.
2. **Clean Data**: Preprocess and remove unnecessary columns.
3. **Train Model**: Train the model using MLflow for tracking.
4. **Evaluate Model**: Evaluate the model and log metrics using MLflow.

### Deployment Pipeline
1. **Deployment Trigger**: Check if the model meets the deployment criteria (e.g., MSE threshold).
2. **Model Deployment**: Deploy the model using MLflow if criteria are met.

## 🖥️ Running the Pipelines

- **Training Pipeline**:
  ```bash
  python run_pipeline.py
  ```

- **Continuous Deployment Pipeline**:
  ```bash
  python run_deployment.py
  ```

## 🎥 Streamlit Demo

A live demo of the project is available [here](https://share.streamlit.io/ayush714/customer-satisfaction/main). To run the Streamlit app locally:

```bash
streamlit run streamlit_app.py
```

## 📚 Resources

- **Blog Post**: [Predicting Customer Satisfaction Before Purchase](https://blog.zenml.io/customer_satisfaction/)
- **Video Tutorial**: [Watch on YouTube](https://youtu.be/L3_pFTlF9EQ)

## ❓ FAQ

1. **Error: `No Step found for the name mlflow_deployer`**  
   **Solution**: Delete the artifact store and rerun the pipeline. Locate the artifact store with:
   ```bash
   zenml artifact-store describe
   ```
   Then delete it:
   ```bash
   rm -rf PATH
   ```

2. **Error: `No Environment component with name mlflow is currently registered`**  
   **Solution**: Install the MLflow integration:
   ```bash
   zenml integration install mlflow -y
   ```

---

This version is concise, professional, and easy to follow. It highlights the key aspects of the project while providing clear instructions for setup and troubleshooting.
