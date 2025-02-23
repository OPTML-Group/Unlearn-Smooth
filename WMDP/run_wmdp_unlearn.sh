#!/bin/bash

set -e

echo "Starting unlearning model experiments..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO.json \
    --unlearn.lr=5e-06 \
    --unlearn.NPO+FT.beta=0.0225 \
    --unlearn.NPO+FT.gamma=1.0 \
    --overall.seed=1001 \
    --unlearn.max_steps=125

echo "Finished: NPO+FT"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO+SAM.json \
    --unlearn.lr=5e-06 \
    --unlearn.NPO+FT+SAM.beta=0.015 \
    --unlearn.NPO+FT+SAM.gamma=2.25 \
    --unlearn.NPO+FT+SAM.sam_rho=0.01 \
    --overall.seed=1001 \
    --unlearn.max_steps=125

echo "Finished: NPO+FT+SAM"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO+RS.json \
    --unlearn.lr=1e-05 \
    --unlearn.NPO+FT+RS.beta=0.0125 \
    --unlearn.NPO+FT+RS.gamma=2.0 \
    --unlearn.NPO+FT+RS.sam_rho=0.0001 \
    --overall.seed=1001 \
    --unlearn.max_steps=125

echo "Finished: NPO+FT+RS"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO.json \
    --unlearn.unlearn_method NPO+FT+CR \
    --logger.json.root files/results/unlearn_wmdp_bio/NPO+CR \
    --unlearn.lr=8.5e-06 \
    --unlearn.NPO+FT+CR.beta=0.035 \
    --unlearn.NPO+FT+CR.gamma=1.0 \
    --unlearn.NPO+FT+CR.gnr_rho=10.0 \
    --overall.seed=1001 \
    --unlearn.max_steps=125

echo "Finished: NPO+FT+CR"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO.json \
    --unlearn.unlearn_method NPO+FT+GNR \
    --logger.json.root files/results/unlearn_wmdp_bio/NPO+GNR \
    --unlearn.lr=7.5e-06 \
    --unlearn.NPO+FT+GNR.beta=0.04 \
    --unlearn.NPO+FT+GNR.gamma=1.0 \
    --unlearn.NPO+FT+GNR.gnr_rho=10.0 \
    --overall.seed=1001 \
    --unlearn.max_steps=125

echo "Finished: NPO+FT+GP"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO.json \
    --logger.json.root files/results/unlearn_wmdp_bio/NPO+SWA \
    --unlearn.lr=5.5e-06 \
    --unlearn.NPO+FT.beta=0.05 \
    --unlearn.NPO+FT.gamma=2.0 \
    --overall.seed=1001 \
    --unlearn.max_steps=125 \
    --unlearn.swa 1

echo "Finished: NPO+WA"

echo "All experiments completed successfully!"
