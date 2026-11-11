"""Record the authors' batch settings and selected examples without changing training."""

import argparse
import json
import runpy
import sys
from pathlib import Path

import torch
from transformers import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--entry", choices=["unlearn", "relearn"], default="unlearn")
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()
    # Release the training process before the authors' lm-eval subprocesses load weights.
    # The shell runner evaluates the saved checkpoint with the same task settings.
    sys.path.insert(0, str(Path.cwd() / "src"))
    from importlib import import_module

    module = import_module(f"model.{args.entry}")
    trainer_class = module.Unlearn if args.entry == "unlearn" else module.Relearn
    trainer_class.eval = lambda self, logger: None
    original_train = Trainer.train

    def train(trainer, *positional, **keywords):
        parameters = list(trainer.model.parameters())
        batch = trainer.args.train_batch_size
        accumulation = trainer.args.gradient_accumulation_steps
        report = {
            "hf_device_map": trainer.model.hf_device_map,
            "is_model_parallel": trainer.is_model_parallel,
            "n_gpu": trainer.args.n_gpu,
            "world_size": trainer.args.world_size,
            "per_device_batch_size": trainer.args.per_device_train_batch_size,
            "train_batch_size": batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": batch * accumulation * trainer.args.world_size,
            "dataset_size": len(trainer.train_dataset),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "parameter_devices": sorted({str(p.device) for p in parameters}),
            "max_steps": trainer.args.max_steps,
            "warmup_steps": trainer.args.warmup_steps,
            "learning_rate": trainer.args.learning_rate,
            "lr_scheduler_type": trainer.args.lr_scheduler_type.value,
            "max_grad_norm": trainer.args.max_grad_norm,
            "seed": trainer.args.seed,
            "data_seed": trainer.args.data_seed,
            "bf16": trainer.args.bf16,
        }
        if hasattr(trainer.train_dataset, "indices"):
            report["selected_indices"] = list(trainer.train_dataset.indices)
        if args.expected_steps is not None:
            assert trainer.args.max_steps == args.expected_steps, report
        Path(args.report).write_text(json.dumps(report, indent=2))
        assert trainer.args.world_size == 1
        assert report["effective_batch_size"] == 4, report
        result = original_train(trainer, *positional, **keywords)
        report["completed_steps"] = trainer.state.global_step
        report["training_loss"] = result.training_loss
        report["completed_epochs"] = trainer.state.epoch
        report["log_history"] = trainer.state.log_history
        report["training_seconds"] = result.metrics["train_runtime"]
        report["peak_cuda_allocated_gib"] = {
            str(device): torch.cuda.max_memory_allocated(device) / 2**30
            for device in {p.device for p in parameters if p.is_cuda}
        }
        Path(args.report).write_text(json.dumps(report, indent=2))
        assert trainer.state.global_step == trainer.args.max_steps
        return result

    Trainer.train = train
    try:
        entry = f"src/exec/{args.entry}_model.py"
        sys.argv = [entry, "--config-file", args.config]
        runpy.run_path(entry, run_name="__main__")
    finally:
        Trainer.train = original_train


if __name__ == "__main__":
    main()
