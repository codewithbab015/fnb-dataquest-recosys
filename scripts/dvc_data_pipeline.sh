#!/bin/bash
set -e

# Configuration
VENV_PATH="/mnt/d/research-workspace/workx-projects/.venv"
RAW_FILE="dq_recsys_challenge_2025(in).csv"
PROCESS_FILE="processed_fnb.csv"
TRAIN_FILE="train.csv"
TEST_FILE="test.csv"
MODEL_OUTPUT="models/classifier"

# Activate virtual environment
echo "Activating Python virtual environment..."
source "${VENV_PATH}/bin/activate"

# Stage 1: Process raw data
echo "Running data processing stage..."
dvc stage add --force --name process \
    --deps "src/engineering/process.py" \
    --deps "data/raw/${RAW_FILE}" \
    --outs "data/processed/${PROCESS_FILE}" \
    python src/engineering/process.py --raw "'data/raw/${RAW_FILE}'" --process "data/processed/${PROCESS_FILE}"

# Stage 2: Feature engineeringing
echo "Running feature engineeringing stage..."
dvc stage add --force --name feature \
    --deps "src/engineering/feature.py" \
    --deps "data/processed/${PROCESS_FILE}" \
    --outs "data/training/${TRAIN_FILE}" \
    --outs "data/training/${TEST_FILE}" \
    python src/engineering/feature.py --process "data/processed/${PROCESS_FILE}" --train "data/training/${TRAIN_FILE}" --test "data/training/${TEST_FILE}"

# Stage 3: Model Training
echo "Running model training stage..."
dvc stage add --force --name trainer \
    --params src/training/model.params.yml:train.mode,train.model\
    --deps "src/training/run_trainer.py" \
    --deps "data/training/${TRAIN_FILE}" \
    --outs "${MODEL_OUTPUT}" \
    python src/training/run_trainer.py --data-path "data/training/${TRAIN_FILE}" --output "${MODEL_OUTPUT}"

# Cleanup
echo "Deactivating Python virtual environment..."
deactivate

echo "Pipeline stages added successfully!"