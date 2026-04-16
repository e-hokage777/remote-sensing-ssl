#!/bin/bash

# Script to extract all features, for all SSL methods for a particular dataset
DATASET_ID=$1

if [ -d "data/features/$DATASET_ID" ]; then
    echo "Directory exists"
    exit 1
else
    mkdir data/features/$DATASET_ID
fi

for method in "simclr" "moco" "byol" "barlow"; do
    for backbone_data in "eurosat" "resisc45"; do
        echo "Extracting features for $method and $backbone_data"
        python extract_features.py --dataset-id $DATASET_ID --method $method --backbone-data $backbone_data --export-csv "$DATASET_ID-$method-$backbone_data.csv" --output-dir "data/features/$DATASET_ID/"
    done
done

echo "Extraction Complete"
