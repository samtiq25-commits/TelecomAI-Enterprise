import joblib,pandas as pd
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from .config import ANOMALY_MODEL,FAILURE_MODEL
NF=["network_traffic","latency","packet_loss","signal_strength","cpu_usage","memory_usage","call_drop_rate"]
EF=["temperature","uptime","cpu_usage","memory_usage","error_count","maintenance_count"]
def train_anomaly_model(df):
    m=IsolationForest(n_estimators=200,contamination=.03,random_state=42); m.fit(df[NF]); joblib.dump(m,ANOMALY_MODEL); return m
def detect_anomalies(df):
    m=joblib.load(ANOMALY_MODEL) if ANOMALY_MODEL.exists() else train_anomaly_model(df)
    out=df.copy(); out["anomaly_score"]=m.decision_function(out[NF]); out["is_anomaly"]=m.predict(out[NF])==-1; return out
def train_failure_model(df):
    m=RandomForestClassifier(n_estimators=250,max_depth=8,class_weight="balanced",random_state=42); m.fit(df[EF],df["failure"]); joblib.dump(m,FAILURE_MODEL); return m
def predict_failures(df):
    m=joblib.load(FAILURE_MODEL) if FAILURE_MODEL.exists() else train_failure_model(df)
    out=df.copy(); out["failure_probability"]=m.predict_proba(out[EF])[:,1]
    out["risk_level"]=pd.cut(out["failure_probability"],[-.01,.3,.6,.8,1.01],labels=["LOW","MEDIUM","HIGH","CRITICAL"])
    return out.sort_values("failure_probability",ascending=False)
