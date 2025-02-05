import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Procesamiento de recetas con modelos de ML")
    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Ruta del archivo JSON de recetas de entrada"
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Ruta del archivo JSON donde se guardarán las recetas procesadas"
    )
    parser.add_argument(
        "--preprocessing_strategy", type=str, choices=["NLKT", "Gensim", "SpaCy"], required=True,
        help="Libería utilizada para el preprocesado de los textos mediante la implementación de un pipeline"
    )
    parser.add_argument(
        "--vectorization", type=str, choices=["tfidf", "Word2Vec", "Contextual_Embedding"], default="tfidf",
        help="Método de vectorización a utilizar (TF-IDF, Word2Vec o Transformer embeddings BERT/RoBERTa)"
    )
    parser.add_argument(
        "--ml_strategy", type=str, choices=["neural_networks", "knn", "cnn", "svm", "cart", "random_forest"], required=True,
        help="Estrategia de aprendizaje automático a emplear (Redes neuronales, K-NN, CNN, SVM, CART, Random Forest, etc.)"
    )
    parser.add_argument(
        "--use_kafka", action="store_true",
        help="(Opcional) Indica si se debe usar Kafka para ingestar nuevas recetas en tiempo real"
    )
    parser.add_argument(
        "--use_mongodb", action="store_true",
        help="(Opcional) Indica si se quiere almacenar los resultados en MongoDB"
    )
    parser.add_argument(
        "--use_llama", action="store_true",
        help="(Opcional) Utilizar LLAMa para embeddings o generación de texto"
    )
    parser.add_argument(
        "--use_mixtral", action="store_true",
        help="(Opcional) Utilizar Mixtral para generación o procesamiento de texto"
    )
    parser.add_argument(
        "--use_extractktl", action="store_true",
        help="(Opcional) Usar ExtractKTL para extracción de información clave"
    )
    parser.add_argument(
        "--use_gensim", action="store_true",
        help="(Opcional) Utilizar Gensim para procesamiento de texto"
    )
    parser.add_argument(
        "--use_nltk", action="store_true",
        help="(Opcional) Utilizar NLTK para análisis y procesamiento de lenguaje natural"
    )
    parser.add_argument(
        "--use_spacy", action="store_true",
        help="(Opcional) Utilizar SpaCy para análisis y procesamiento de lenguaje natural"
    )
    parser.add_argument(
        "--use_pydantic", action="store_true",
        help="(Opcional) Utilizar Pydantic para validación de datos"
    )
    parser.add_argument(
        "--use_grafana", action="store_true",
        help="(Opcional) Utilizar Grafana para visualización de datos"
    )
    parser.add_argument(
        "--use_powerbi", action="store_true",
        help="(Opcional) Utilizar Power BI para visualización de datos"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
