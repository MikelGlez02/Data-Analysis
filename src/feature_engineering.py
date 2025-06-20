import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
from typing import Tuple, Union

class FeatureEngineer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.bert_tokenizer = None
        self.bert_model = None
    
    def tfidf_features(self, texts: pd.Series, max_features: int = 5000) -> pd.DataFrame: # Características de TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
    
    def word2vec_features(self, texts: pd.Series, vector_size: int = 100, window: int = 5, min_count: int = 2) -> pd.DataFrame: # Generamos características W2V

        tokenized_texts = [text.split() for text in texts] # Tokenizar textos
        self.word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4) # Se entrena el modelo W2V
        features = [] # Agrupamos las características según el promedio de los vectores de palabras
        for text in tokenized_texts:
            vectors = [self.word2vec_model.wv[word] for word in text if word in self.word2vec_model.wv]
            if len(vectors) > 0:
                features.append(np.mean(vectors, axis=0))
            else:
                features.append(np.zeros(vector_size))
        
        columns = [f'w2v_{i}' for i in range(vector_size)]
        return pd.DataFrame(features, columns=columns)
    
    def bert_features(self, texts: pd.Series, model_name: str = 'bert-base-uncased') -> pd.DataFrame: # Se generan características del modelo BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name) # Cargamos modelo y tokenizer
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.eval() # Configuramos el modelo para la evaluación
        features = []
        for text in texts:
            inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length') # Tokenizamos el texto
            with torch.no_grad():
                outputs = self.bert_model(**inputs) # Embeddings
            last_hidden_state = outputs.last_hidden_state 
            mean_embedding = torch.mean(last_hidden_state, dim=1).squeeze().numpy() # Embedding promedio de la última capa
            features.append(mean_embedding)
        columns = [f'bert_{i}' for i in range(features[0].shape[0])]
        return pd.DataFrame(features, columns=columns)
    
    def combine_features(self, text_features: pd.DataFrame, numeric_features: pd.DataFrame = None) -> pd.DataFrame: # Combinación entre características textuales y numéricas
        if numeric_features is not None:
            return pd.concat([text_features, numeric_features], axis=1)
        return text_features
    
    def reduce_dimensions(self, features: pd.DataFrame, n_components: int = 50) -> pd.DataFrame: # Análisis de componentes principales (PCA) para reducir la dimensionalidad
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
        columns = [f'pca_{i}' for i in range(n_components)]
        return pd.DataFrame(reduced_features, columns=columns)
