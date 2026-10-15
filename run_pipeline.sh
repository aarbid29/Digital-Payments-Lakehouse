#!/usr/bin/env bash
# Rebuilds everything from scratch.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf data warehouse
python src/generate_data.py
python src/bronze.py
python src/silver.py
python src/quality.py --stage silver
python src/gold.py
python src/quality.py --stage gold
python src/export_serving.py
python src/train_segmentation.py
python src/build_churn_features.py
python src/train_churn.py
python -m pytest -q
echo "Pipeline complete."