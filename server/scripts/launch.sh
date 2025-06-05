#!/bin/bash

# Set working directory to project root
cd "$(dirname "$0")/.." || exit

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found in .venv directory"
    exit 1
fi

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export "$(cat .env | xargs)"
fi

# Set Python path
export PYTHONPATH="${PWD}"

# Run the FastAPI application using fastapi cli
fastapi dev ./app/main.py