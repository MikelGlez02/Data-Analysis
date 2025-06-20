import argparse
import pandas as pd
from preprocess import load_data, preprocess_data
from feature_engineering import FeatureEngineer
from model import RegressionModel
from summarizer import load_summarizer
import json
from typing import Dict, Any
import os

def save_results(results: Dict[str, Any], filename: str = 'results.json'):
    """Guardamos los resultados en un archivo JSON"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Proyecto Final: Tratamiento de Datos')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help='Modo de ejecución')
    parser.add_argument('--data', type=str, required=True, choices=['AllData', 'OnlyTextData', 'OnlySomeTextData', 'Directions', 'Descriptions'], help='Configuración de datos a usar')
    parser.add_argument('--model', type=str, required=False, choices=['KNN', 'SVM', 'RF', 'NN', 'XGB', 'LGBM', 'CART', 'BERT'], help='Modelo a entrenar/evaluar')
    parser.add_argument('--summarizer', type=str, required=False, choices=['BART', 'T5', 'LED', 'PEGASUS'],  help='Modelo de resumen a usar')
    args = parser.parse_args()
    
    print("Cargando datos...")
    df = load_data('data/full_format_recipes.json') # Cargamos datos
    print("Preprocesando datos...")
    df = preprocess_data(df, args.data) # preprocesamiento de datos
    fe = FeatureEngineer() # Función de ingeniería de características (Feature Emgineering)
    print("Generando características de texto...")
    if args.model == 'BERT':
        text_features = fe.bert_features(df['processed_text'])
    else:
        text_features = fe.tfidf_features(df['processed_text']) # Usar TF-IDF para modelos tradicionales
    
    if args.data in ['AllData', 'OnlyTextData']: # Combinar con características numéricas si es necesario
        numeric_features = df[['calories', 'fat', 'protein', 'sodium']]
        features = fe.combine_features(text_features, numeric_features)
    else:
        features = text_features

    if features.shape[1] > 100:  # Reducir dimensionalidad si hay muchas características
        features = fe.reduce_dimensions(features)
    
    target = df['rating']
    rm = RegressionModel()  # Inicializar modelo de regresión
    if args.mode == 'train':
        print(f"Entrenando modelo {args.model}...")
        X_train, X_test, y_train, y_test = rm.prepare_data(features, target) # Preparar datos
        if args.model == 'BERT': # Entrenar modelo según el tipo
            model = rm.train_bert(X_train, y_train)
        else:
            model = rm.train_model(args.model, X_train, y_train)
        
        metrics = rm.evaluate_model(model, X_test, y_test) # Evaluar modelo
        print(f"Métricas de evaluación: {metrics}")
        results = {'model': args.model, 'data_config': args.data, 'metrics': metrics}
        save_results(results)
        
    elif args.mode == 'evaluate':
        print(f"Evaluando modelo {args.model} con validación cruzada...")
        cv_results = rm.cross_validate(args.model, features, target)
        print(f"Resultados de validación cruzada: {cv_results}")
        results = {'model': args.model,'data_config': args.data,'cv_metrics': cv_results}
        save_results(results)
    
    if args.summarizer:
        print(f"Cargando resumidor {args.summarizer}...")
        summarizer = load_summarizer(args.summarizer)
        sample_recipes = df.sample(5).to_dict('records') # Seleccionar algunas recetas para resumir
        print("Generando resúmenes...")
        summaries = summarizer.batch_summarize(sample_recipes)
        print("\nEjemplos de resúmenes:")
        for i, (recipe, summary) in enumerate(zip(sample_recipes, summaries)):
            print(f"\nReceta {i+1}: {recipe.get('title', 'Sin título')}")
            print(f"Resumen: {summary}")
        
        # Guardar resúmenes
        summary_results = {'summarizer_model': args.summarizer,'summaries': [ {'title': r.get('title', ''), 'summary': s} for r, s in zip(sample_recipes, summaries) ] }
        save_results(summary_results, 'summaries.json')

if __name__ == '__main__':
    main()
