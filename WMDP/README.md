# WMDP

## Installation

To create a conda environment for Python 3.9, run:
```bash
conda env create -f environment.yml
conda activate wmdp
```

## Get the data
Follow the [link](https://github.com/centerforaisafety/wmdp?tab=readme-ov-file) to download the WMDP-Bio dataset and place it in the `./WMDP/files/data`.

## Get the unlearned model
1. Run the command `./run_unlearn_experiments.sh`.
2. After the command is complete, the checkpoints and results will be stored in `./WMDP/files/results/unlearn_wmdp_bio/NPO+xxx`.

## Relearn attack
1. Run the following command.
    ```bash
    # Same data, different epochs. Use unlearn.num_epochs to change the relearning epochs.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/relearn_model.py \
        --config-file configs/unlearn/wmdp/Relearn+Forget.json \
        --overall.model_name {the path of unlearned model} \
        --unlearn.max_steps=-1 --unlearn.num_epochs 3

    # Different data, same epoch. Use unlearn.max_steps to change the relearning data samples.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/relearn_model.py \
        --config-file configs/unlearn/wmdp/Relearn+Forget.json \
        --overall.model_name {the path of unlearned model} \
        --unlearn.max_steps=15
    ```
2. After the command is complete, the checkpoints and results will be stored in `./WMDP/files/results/unlearn_wmdp_bio/Relearn+Forget`.

## Jailbreak attack
1. `git clone git@github.com:ethz-spylab/unlearning-vs-safety.git`.
2. Add the config of unlearned model in the `model_configs` of `unlearning-vs-safety/flrt_repo/flrt/util.py`. See the following example.
    ```bash
    "The name of unlearned model": ModelConfig(
        model_name="The path of unlearned model",
        peft_path=None,
        response_template="<|assistant|>\n",
        first_token="",
        system_prompt=None,
        sep=" ",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    ),
    ```
3. Run the following command.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python -m src.enhanced_gcg.flrt_repo.demo \
        --model_name_or_path {the path of unlearned model} \
        --optimize_prompts 0,2,3,4,5 \
        --wmdp_subset wmdp-bio \
        --use_static_representations \
        --dont_clamp_loss \
        --attack_layers 20 \
        --use_init npo-bio \
        --max_iter 1500
    ```