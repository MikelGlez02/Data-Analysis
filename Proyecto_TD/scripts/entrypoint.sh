# entrypoint.sh
#!/bin/bash
set -e

# Validar dependencias
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python before proceeding."
    exit 1
fi
if ! command -v pytest &> /dev/null; then
    echo "Pytest is not installed. Please add it to your dependencies."
    exit 1
fi

# Parse the command (e.g., preprocess, train, evaluate, generate_new_recipes)
COMMAND=$1
shift

case "$COMMAND" in
  preprocess)
    echo "Running preprocess..."
    python main.py preprocess "$@"
    ;;
  train)
    echo "Running training..."
    python main.py train "$@"
    ;;
  evaluate)
    echo "Running evaluation..."
    python main.py evaluate "$@"
    ;;
  generate_new_recipes)
    echo "Generating new recipes..."
    python main.py generate_new_recipes "$@"
    ;;
  test)
    echo "Running tests..."
    pytest tests/ --disable-warnings
    ;;
  *)
    echo "Invalid command: $COMMAND"
    exit 1
    ;;
esac
