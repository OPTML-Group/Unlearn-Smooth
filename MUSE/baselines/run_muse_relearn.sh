#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Starting relearning experiments..."

# Define model paths and data files
BOOKS_MODEL="./baselines/ckpt/books/sam_npo_gdr/checkpoint-139"
NEWS_MODEL="./baselines/ckpt/news/sam_npo_gdr/checkpoint-1020"
TOKENIZER="meta-llama/Llama-2-7b-hf"
BOOKS_DATA="../data/books/raw/forget.txt"
NEWS_DATA="../data/news/raw/forget.txt"

# Define max_steps values for books and news separately
BOOKS_MAX_STEPS=(50 75 100)
NEWS_MAX_STEPS=(100 125 150)

# Run experiments for books
for steps in "${BOOKS_MAX_STEPS[@]}"; do
    echo "Running: Books Relearning with max_steps=$steps..."
    python relearn.py \
        --model_dir "$BOOKS_MODEL" \
        --tokenizer_dir "$TOKENIZER" \
        --data_file "$BOOKS_DATA" \
        --max_len 2048 \
        --max_steps "$steps" \
        --lr 1e-5 \
        --per_device_batch_size 4
    echo "Finished: Books Relearning with max_steps=$steps"
done

# Run experiments for news
for steps in "${NEWS_MAX_STEPS[@]}"; do
    echo "Running: News Relearning with max_steps=$steps..."
    python relearn.py \
        --model_dir "$NEWS_MODEL" \
        --tokenizer_dir "$TOKENIZER" \
        --data_file "$NEWS_DATA" \
        --max_len 2048 \
        --max_steps "$steps" \
        --lr 1e-5 \
        --per_device_batch_size 4
    echo "Finished: News Relearning with max_steps=$steps"
done

echo "All relearning experiments completed successfully!"
