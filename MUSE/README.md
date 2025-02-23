# MUSE

## Installation

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate muse
```

## Get the data & origin models

- Two corpora `News` and `Books` and the associated target models are available as follows:
    | Domain | <div style="text-align: center">Target Model for Unlearning</div> | Dataset |
    |----------|:------------------------------:|----------| 
    | News | [Target model](https://huggingface.co/muse-bench/MUSE-News_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-News) |
    | Books | [Target model](https://huggingface.co/muse-bench/MUSE-Books_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-Books) | 

- Before proceeding, load all the data from HuggingFace to the root of this repostiory by running the following instruction:
    ```
    python load_data.py
    ```

## Get the unlearned model
1. Run `run_muse_unlearn.sh` in the `baselines` folder.
    - `algo`: Unlearning algorithm to run (`npo_gdr`, `sam_npo_gdr`).
    - `model_dir`: Directory of the target model.
    - `tokenizer_dir`: Directory of the tokenizer.
    - `data_file`: Forget set.
    - `retain_data_file`: Retain set for GDR/KLR regularizations if required by the algorithm.
    - `out_dir`: Directory to save the unlearned model (default: `ckpt`).
    - `max_len`: Maximum input length (default: 2048).
    - `per_device_batch_size`, `epochs`, `lr`: Hyperparameters.

2. Resulting models are saved in the `ckpt` folder as shown:
    ```
    ckpt
    ├── news/
    │   ├── sam_npo_gdr/
    │   │   ├── checkpoint-102
    │   │   ├── checkpoint-204
    │   │   ├── checkpoint-306
    │   │   └── ...
    │   └── npo/
    │       └── ...
    └── books/
        ├── sam_npo_gdr
        └── ...
    ```

## Get the relearned model
- Run `run_muse_relearn.sh` in the `baselines` folder.


## Evaluate the unlearned model

- To evaluate your unlearned model(s), run `eval.py` from the root of this repository with the following command-line arguments:

    - `--model_dirs`: A list of directories containing the unlearned models. These can be either HuggingFace model directories or local storage paths.
    - `--names`: A unique name assigned to each unlearned model in `--model_dirs`. The length of `--names` should match the length of `--model_dirs`.
    - `--corpus`: The corpus to use for evaluation. Options are `news` or `books`.
    - `--out_file`: The name of the output file. The file will be in CSV format, with each row corresponding to an unlearning method from `--model_dirs`, and columns representing the metrics specified by `--metrics`.
    - `--tokenizer_dir` (Optional): The directory of the tokenizer. Defaults to `meta-llama/Llama-2-7b-hf`, which is the default tokenizer for LLaMA.
    - `--metrics` (Optional): The metrics to evaluate. Options are `verbmem_f` (VerbMem Forget), `privleak` (PrivLeak), `knowmem_f` (KnowMem Forget), and `knowmem_r` (Knowmem Retain, i.e., Utility). Defaults to evaluating all these metrics.
    - `--temp_dir` (Optional): The directory for saving intermediate computations. Defaults to `temp`.

- Run the following command with placeholder values:

    ```python
    python eval.py \
    --model_dirs "repo/model1" "repo/model2" \
    --names "model1" "model2" \
    --corpus books \
    --out_file "out.csv"
    ```

- For `News`, we select the result of NPO+SAM from epoch 10. For `Books`, we select the result of NPO+SAM from epoch 1.