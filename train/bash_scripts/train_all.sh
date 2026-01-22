#!/bin/bash
# Training script for ALL VTON Models - Mixed Difficulty
# Trains: CATVTON, IDM-VTON, CP-VTON, VTON-GAN, OOTDiffusion, DiT
# Dataset: Configurable difficulty (default: medium)

DIFFICULTY=${1:-medium}  # Default to medium if not specified

echo "=========================================="
echo "Training ALL VTON Models - Mixed Difficulty"
echo "Difficulty: $DIFFICULTY"
echo "Models: CATVTON, IDM-VTON, CP-VTON, VTON-GAN, OOTDiffusion, DiT"
echo "=========================================="

# Define S3 prefixes based on difficulty
if [ "$DIFFICULTY" == "easy" ]; then
    S3_PREFIXES="dataset_ultimate/easy/female/ dataset_ultimate/easy/male/"
elif [ "$DIFFICULTY" == "medium" ]; then
    S3_PREFIXES="dataset_ultimate/easy/female/ dataset_ultimate/easy/male/ dataset_ultimate/medium/female/ dataset_ultimate/medium/male/"
elif [ "$DIFFICULTY" == "hard" ]; then
    S3_PREFIXES="dataset_ultimate/easy/female/ dataset_ultimate/easy/male/ dataset_ultimate/medium/female/ dataset_ultimate/medium/male/ dataset_ultimate/hard/female/ dataset_ultimate/hard/male/"
else
    echo "Invalid difficulty: $DIFFICULTY. Use 'easy', 'medium', or 'hard'."
    exit 1
fi

echo "S3 Prefixes: $S3_PREFIXES"
echo ""

# Train each model
echo "Step 1/6: Training CATVTON..."
bash train_catvton.sh $DIFFICULTY

echo ""
echo "Step 2/6: Training IDM-VTON..."
bash train_idmvton.sh $DIFFICULTY

echo ""
echo "Step 3/6: Training CP-VTON..."
bash train_cpvton.sh $DIFFICULTY

echo ""
echo "Step 4/6: Training VTON-GAN..."
bash train_vtongan.sh $DIFFICULTY

echo ""
echo "Step 5/6: Training OOTDiffusion..."
bash train_ootdiffusion.sh $DIFFICULTY

echo ""
echo "Step 6/6: Training DiT..."
bash train_dit.sh $DIFFICULTY

echo ""
echo "=========================================="
echo "All models training completed!"
echo "Difficulty: $DIFFICULTY"
echo "=========================================="
