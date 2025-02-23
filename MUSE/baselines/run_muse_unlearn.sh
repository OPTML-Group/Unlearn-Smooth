#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Starting unlearning experiments..."

# NPO w/o SAM on News
echo "Running: News NPO without SAM..."
python unlearn.py \
    --algo npo_gdr \
    --model_dir muse-bench/MUSE-News_target \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --data_file ../data/news/raw/forget.txt \
    --retain_data_file ../data/news/raw/retain1.txt \
    --out_dir ./ckpt/news/npo_gdr \
    --max_len 2048 \
    --epochs 10 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
    --beta 0.1 \
    --coeff 1.0 \
    --npo_coeff 1.0

echo "Finished: News NPO without SAM"

# NPO w/o SAM on Books
echo "Running: Books NPO without SAM..."
python unlearn.py \
    --algo npo_gdr \
    --model_dir muse-bench/MUSE-Books_target \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --data_file ../data/books/raw/forget.txt \
    --retain_data_file ../data/books/raw/retain1.txt \
    --out_dir ./ckpt/books/npo_gdr \
    --max_len 2048 \
    --epochs 10 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
    --beta 0.1 \
    --coeff 1.0 \
    --npo_coeff 1.0

echo "Finished: Books NPO without SAM"

# NPO w/ SAM on News
echo "Running: News NPO with SAM..."
python unlearn.py \
    --algo sam_npo_gdr \
    --model_dir muse-bench/MUSE-News_target \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --data_file ../data/news/raw/forget.txt \
    --retain_data_file ../data/news/raw/retain1.txt \
    --out_dir ./ckpt/news/sam_npo_gdr_rho0.01_coeff0.1 \
    --max_len 2048 \
    --epochs 10 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
    --beta 0.1 \
    --coeff 1.0 \
    --npo_coeff 1.0 \
    --sam_rho 0.01

echo "Finished: News NPO with SAM"

# NPO w/ SAM on Books
echo "Running: Books NPO with SAM..."
python unlearn.py \
    --algo sam_npo_gdr \
    --model_dir muse-bench/MUSE-Books_target \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --data_file ../data/books/raw/forget.txt \
    --retain_data_file ../data/books/raw/retain1.txt \
    --out_dir ./ckpt/books/sam_npo_gdr \
    --max_len 2048 \
    --epochs 10 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
    --beta 0.1 \
    --coeff 2.5 \
    --npo_coeff 1.0 \
    --sam_rho 0.01

echo "Finished: Books NPO with SAM"

echo "All experiments completed successfully!"
