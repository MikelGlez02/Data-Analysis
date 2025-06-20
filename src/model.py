import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Any
import xgboost as xgb
import lightgbm as lgb

class RegressionDataset(Dataset): # Regresión con PyTorch
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return { 'features': torch.tensor(self.features.iloc[idx].values, dtype=torch.float32), 'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.float32) }

class RegressionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {'KNN': KNeighborsRegressor(),'SVM': SVR(),'RF': RandomForestRegressor(),'NN': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500),'XGB': xgb.XGBRegressor(),'LGBM': lgb.LGBMRegressor(),'CART': DecisionTreeRegressor()}
        self.best_model = None
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2) -> Tuple: # Estandarizar características
        features_scaled = self.scaler.fit_transform(features)
        features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=test_size, random_state=42) # Dividimos train y test
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any: # Entrenamos el modelo específico
        if model_name not in self.models:
            raise ValueError(f"Modelo no soportado: {model_name}")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict: # Evaluación del modelo para que devuelva las métricas
        predictions = model.predict(X_test)
        return { 'mse': mean_squared_error(y_test, predictions),'mae': mean_absolute_error(y_test, predictions),'r2': r2_score(y_test, predictions) }
    
    def cross_validate(self, model_name: str, features: pd.DataFrame, target: pd.Series, cv: int = 5) -> Dict: # Validación cruzada
        if model_name not in self.models:
            raise ValueError(f"Modelo no soportado: {model_name}")
        model = self.models[model_name]
        pipeline = Pipeline([ ('scaler', StandardScaler()), ('model', model) ])
        scores = {'mse': cross_val_score(pipeline, features, target, cv=cv, scoring='neg_mean_squared_error') * -1,'mae': cross_val_score(pipeline, features, target, cv=cv, scoring='neg_mean_absolute_error') * -1, 'r2': cross_val_score(pipeline, features, target, cv=cv, scoring='r2') }
        return {'mean_mse': np.mean(scores['mse']),'std_mse': np.std(scores['mse']),'mean_mae': np.mean(scores['mae']),'std_mae': np.std(scores['mae']),'mean_r2': np.mean(scores['r2']),'std_r2': np.std(scores['r2']) }
    
    def train_bert(self, features: pd.DataFrame, target: pd.Series) -> Any: # Modelo BERT con Fine-Tuning
        train_dataset = RegressionDataset(features, target) # Dataset de Regresión
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1, problem_type="regression") # Modelo BERT de Regresión
        training_args = TrainingArguments( # Configuramos argumentos de entrenamiento
            output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01, logging_dir='./logs', logging_steps=10,
        )
        trainer = Trainer( model=model, args=training_args,train_dataset=train_dataset,) # Se crea Trainer
        trainer.train() # Entrenamos el modelo
        return trainer
