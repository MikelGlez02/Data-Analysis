import json
import pandas as pd
import spacy
import re
from typing import Dict, List, Union
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str: # Limpia y normaliza el texto
        if not isinstance(text, str):
            return ""
        text = text.lower()  # A minúsculas
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Elimina carácteres especial y/o números
        text = re.sub(r'\s+', ' ', text).strip() # Elimina espacios
        return text
    
    def lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc] # Lematiza el texto (spaCy)
        return ' '.join(lemmas)
    
    def remove_stopwords(self, text: str) -> str:
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words] # Eliminamos palabras vacías del texto
        return ' '.join(filtered_text)
    
    def preprocess_pipeline(self, text: str) -> str: # Pipeline completo de preprocesamiento
        text = self.clean_text(text)
        text = self.lemmatize_text(text)
        text = self.remove_stopwords(text)
        return text

def load_data(file_path: str) -> pd.DataFrame: 
    with open(file_path, 'r', encoding='utf-8') as f: # Carga los datos de un archivo JSON mediante UTF-8
        data = json.load(f)
    return pd.DataFrame(data)

def combine_text_features(df: pd.DataFrame, data_config: str) -> pd.Series: # Combinación de diferentes columnas de texto

    if data_config == "AllData":  # Combinar todas las características de texto con metadatos numéricos
        df['combined_text'] = df['title'].fillna('') + ' ' + \ df['desc'].fillna('') + ' ' + \ df['categories'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' + \ df['directions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        return df['combined_text']
    elif data_config == "OnlyTextData": # Solo características de texto
        df['combined_text'] = df['title'].fillna('') + ' ' + \ df['desc'].fillna('') + ' ' + \ df['categories'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' + \ df['directions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        return df['combined_text']
    elif data_config == "OnlySomeTextData": # Solo instrucciones y descripciones
        df['combined_text'] = df['desc'].fillna('') + ' ' + \  df['directions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        return df['combined_text']
    elif data_config == "Directions":  # Solo instrucciones
        return df['directions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    elif data_config == "Descriptions":
        # Solo descripciones
        return df['desc'].fillna('')
    else:
        raise ValueError(f"Configuración de datos no válida: {data_config}")

def preprocess_data(df: pd.DataFrame, data_config: str) -> pd.DataFrame: # Preprocesamiento de datos

    preprocessor = TextPreprocessor()
    df['processed_text'] = combine_text_features(df, data_config).apply(preprocessor.preprocess_pipeline) # Procesar texto
    numeric_cols = ['rating', 'calories', 'fat', 'protein', 'sodium'] # Limpieza de datos numéricos
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    
    return df
