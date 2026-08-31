#!/bin/bash
# Runs the full pipeline end to end: clean -> features -> SQL load -> train -> evaluate -> business metrics
set -e

echo "1/5 Cleaning data..."
python3 src/data_prep.py

echo -e "\n2/5 Engineering features..."
python3 src/features.py

echo -e "\n3/5 Loading into SQLite..."
python3 src/sql_loader.py

echo -e "\n4/5 Training and comparing models..."
python3 src/train_model.py

echo -e "\n5/5 Generating evaluation plots and business metrics..."
python3 src/evaluate.py
python3 src/business_metrics.py

echo -e "\nDone. Check reports/ and models/ for outputs."
