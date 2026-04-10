
#!/bin/bash 


# Script to extract all features, for all SSL methods for a particular dataset
DATASET_ID=$1

for method in "simclr" "moco" "byol" "barlow"; do
    for backbone_data in "eurosat" "resisc45"; do
        echo "Extracting features for $method and $backbone_data"
        python extract_features.py --dataset_id $DATASET_ID --method $method --backbone_data $backbone_data --export_csv "$DATASET_ID-$method-$MODEL_DATASET.csv"
    done
done

echo "Extraction Complete"
