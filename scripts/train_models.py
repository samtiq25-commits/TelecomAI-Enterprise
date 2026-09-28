from src.data_generator import load_data
from src.ml_models import train_anomaly_model,train_failure_model
n,e,_=load_data(); train_anomaly_model(n); train_failure_model(e); print("Models trained successfully.")
