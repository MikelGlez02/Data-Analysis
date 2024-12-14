# entrypoint.sh
#!/bin/bash
set -e

# Parse the command (e.g., preprocess, train, evaluate, generate_new_recipes)
COMMAND=$1
shift

case "$COMMAND" in
  preprocess)
    python main.py preprocess "$@"
    ;;
  train)
    python main.py train "$@"
    ;;
  evaluate)
    python main.py evaluate "$@"
    ;;
  generate_new_recipes)
    python main.py generate_new_recipes "$@"
    ;;
  test)
    pytest tests/ --disable-warnings
    ;;
  *)
    echo "Invalid command: $COMMAND"
    exit 1
    ;;
esac
