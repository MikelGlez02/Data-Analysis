# main.py
import argparse
import logging
import sys
from utils.arg_parser import parse_arguments
from utils.logger import setup_logger
from utils.version_checker import check_python_version
from models.regression import train_regression_model, evaluate_regression_model
from models.transformers import fine_tune_transformer, evaluate_transformer_model
from preprocessing.data_analysis import analyze_relationships
from preprocessing.text_cleaner import clean_text
from preprocessing.embeddings import get_embeddings
from database.mongodb_handler import MongoDBHandler
import pandas as pd

def main():
    # Verify Python version
    check_python_version()

    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logger(args.log_level)
    logger = logging.getLogger(__name__)

    # Initialize MongoDB handler
    db_handler = MongoDBHandler(database_name="recipes_project", collection_name="recipes")

    # Entry points for different modes
    if args.mode == "preprocess":
        logger.info("Starting data preprocessing...")
        # Load data
        data = pd.read_json(args.input_data)
        
        # Clean text data
        logger.info("Cleaning text data...")
        data["desc_cleaned"] = data["desc"].apply(clean_text)
        
        # Generate embeddings if advanced mode
        if args.preprocess_mode == "advanced":
            logger.info("Generating embeddings...")
            data["desc_embeddings"] = data["desc_cleaned"].apply(get_embeddings)

        # Save processed data
        logger.info("Saving processed data to MongoDB...")
        db_handler.insert_many(data.to_dict(orient="records"))

    elif args.mode == "train":
        logger.info("Starting training process...")
        if args.model_type == "pytorch":
            train_regression_model(
                vectorizer=args.vectorizer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        elif args.model_type == "transformers":
            fine_tune_transformer(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    elif args.mode == "evaluate":
        logger.info("Starting evaluation process...")
        if args.model_type == "pytorch":
            evaluate_regression_model(metric=args.evaluation_metric)
        elif args.model_type == "transformers":
            evaluate_transformer_model(metric=args.evaluation_metric)
    elif args.mode == "analyze":
        logger.info("Starting data analysis...")
        data = pd.DataFrame(db_handler.find_all())
        analyze_relationships(data)
    elif args.mode == "generate_new_recipes":
        logger.info("Generating new recipes...")
        # Call generation function here
    else:
        logger.error(f"Invalid mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

























