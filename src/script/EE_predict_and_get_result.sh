#!/bin/bash

DOMAINS=("ai" "literature" "music" "politics" "science")

YAML_FILE="SAM-NER/src/yaml/EE/EE_predict.yaml"

PY_SCRIPT="SAM-NER/src/EntityExplor/getPredictResults.py"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PY_SCRIPT not found in current directory."
    exit 1
fi

echo "========================================"
echo "Start Batch Prediction and Extraction"
echo "========================================"

for domain in "${DOMAINS[@]}"
do
    echo "[Current Domain]: $domain"

    sed -i "s/^eval_dataset: .*/eval_dataset: $domain/" "$YAML_FILE"

    echo "-> Updated $YAML_FILE with eval_dataset: $domain"

    echo "-> Running inference with llamafactory..."

    CUDA_VISIBLE_DEVICES=0,1,2 llamafactory-cli train "$YAML_FILE"

    echo "-> Extracting results using Python script..."
    python "$PY_SCRIPT" "$domain"

    echo "-> Finished processing $domain"
    echo "----------------------------------------"
done

echo "All domains processed successfully."