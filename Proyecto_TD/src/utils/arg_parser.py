def parse_arguments():
    parser = argparse.ArgumentParser(description="Proyecto Final: Tratamiento de Datos")

    parser.add_argument("--mode", type=str, required=True, choices=["preprocess", "train", "evaluate", "generate_new_recipes"],
        help="Mode of operation: preprocess, train, evaluate, generate_new_recipes."
    )

    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default is INFO."
    )

    # Additional arguments for different modes
    parser.add_argument("--input_data", type=str, help="Path to input data file.")
    parser.add_argument("--output_data", type=str, help="Path to output data file.")
    parser.add_argument("--preprocess_mode", type=str, choices=["basic", "advanced"], help="Preprocessing mode.")
    parser.add_argument("--model_type", type=str, choices=["pytorch", "scikit-learn"], help="Type of model to use.")
    parser.add_argument("--vectorizer", type=str, choices=["tfidf", "word2vec", "transformers"], help="Vectorization method.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training.")
    parser.add_argument("--evaluation_metric", type=str, choices=["mae", "mse", "r2"], help="Evaluation metric.")

    return parser.parse_args()
