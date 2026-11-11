"""Resolve Figure 3 experiments into configs consumed by the authors' entry points."""

import argparse
import json
import shutil
from pathlib import Path


def experiments(source, root, cache, recipe):
    """Build two defenses and paired trials; the shared N=20/M=1 case runs once."""
    records = []
    for method, settings in recipe["methods"].items():
        config = json.loads((source / settings["config"]).read_text())
        config["overall"].update(seed=recipe["defense_seed"], cache_dir=str(cache))
        config["unlearn"].update(
            max_steps=recipe["defense_steps"], lr=recipe["defense_lr"]
        )
        config["unlearn"].update(settings.get("unlearn", {}))
        objective = config["unlearn"]["unlearn_method"]
        config["unlearn"][objective].update(
            {k: v for k, v in settings.items() if k not in {"config", "unlearn"}}
        )
        config["logger"].update(name=method)
        config["logger"]["json"]["root"] = str(root / "models")
        records.append(
            {
                "kind": "defense",
                "key": method,
                "method": method,
                "expected_steps": recipe["defense_steps"],
                "config": config,
            }
        )
        for case in recipe["cases"]:
            samples, epochs = case["samples"], case["epochs"]
            if samples % 4 or (epochs > 1 and samples != 20):
                raise ValueError("Authors' epoch mode requires 20 samples; batch is 4")
            for seed in recipe["trial_seeds"]:
                key = f"{method}-n{samples}-m{epochs}-seed{seed}"
                config = json.loads((source / recipe["attack_config"]).read_text())
                config["overall"].update(
                    model_name=str(root / "models" / method / "checkpoints"),
                    cache_dir=str(cache),
                    seed=seed,
                )
                config["unlearn"].update(
                    max_steps=samples // 4 if epochs == 1 else -1,
                    num_epochs=epochs,
                    lr=recipe["attack_lr"],
                )
                config["logger"].update(name=key)
                config["logger"]["json"]["root"] = str(root / "attacks")
                records.append(
                    {
                        "kind": "attack",
                        "key": key,
                        "method": method,
                        "samples": samples,
                        "epochs": epochs,
                        "seed": seed,
                        "expected_steps": samples // 4 * epochs,
                        "config": config,
                    }
                )
    return records


def prepare(source, root, cache, recipe):
    root.mkdir(parents=True, exist_ok=False)
    directory = root / "configs"
    directory.mkdir(exist_ok=True)
    for name in ("models", "attacks", "logs", "metrics"):
        (root / name).mkdir(exist_ok=True)
    records = experiments(source, root, cache, recipe)
    for record in records:
        path = directory / (record["key"] + ".json")
        path.write_text(json.dumps(record.pop("config"), indent=2))
        record["config_path"] = str(path)
    (root / "experiments.json").write_text(json.dumps(records, indent=2))
    return records


def evaluation_scores(output, expected_model):
    matches = {}
    for folder, task in [("mmlu.json", "mmlu"), ("wmdp.json", "wmdp_bio")]:
        candidates = []
        for path in (output / folder).rglob("results_*.json"):
            data = json.loads(path.read_text())
            args = data["config"]["model_args"]
            if isinstance(args, str):
                args = dict(p.split("=", 1) for p in args.split(",") if "=" in p)
            if Path(args["pretrained"]).resolve() != expected_model.resolve():
                continue
            if data["config"].get("limit") is None and task in data["results"]:
                candidates.append((path, data["results"][task]["acc,none"]))
        assert len(candidates) == 1, (str(output), task, candidates)
        path, accuracy = candidates[0]
        assert 0 <= accuracy <= 1
        matches[task] = (path, accuracy)
    return matches


def finish(root, record):
    """Validate completed training and full evaluations against frozen settings."""
    config = json.loads(Path(record["config_path"]).read_text())
    output = Path(config["logger"]["json"]["root"]) / config["logger"]["name"]
    expected_model = output / "checkpoints"
    destination = root / "metrics" / record["key"]
    destination.mkdir(parents=True, exist_ok=True)
    result = {**record, "metrics": {}}
    runtime = json.loads((destination / "runtime.json").read_text())
    assert runtime["completed_steps"] == record["expected_steps"], runtime
    assert runtime["effective_batch_size"] == 4, runtime
    if record["kind"] == "attack":
        assert runtime["dataset_size"] == record["samples"]
        assert len(set(runtime["selected_indices"])) == record["samples"]
        assert abs(runtime["completed_epochs"] - record["epochs"]) < 1e-6
    for task, (path, accuracy) in evaluation_scores(output, expected_model).items():
        shutil.copy2(
            path, destination / ("mmlu.json" if task == "mmlu" else "wmdp.json")
        )
        result["metrics"][task] = accuracy
    shutil.copy2(output / "config.json", destination / "authors-config.json")
    if record["kind"] == "defense":
        weight_bytes = sum(
            p.stat().st_size for p in expected_model.glob("*.safetensors")
        )
        assert weight_bytes > 10**10, expected_model
        result["weight_bytes"] = weight_bytes
    (destination / "result.json").write_text(json.dumps(result, indent=2))
    return result


def origin(root, model):
    import sys

    sys.path.insert(0, str(Path.cwd() / "src"))
    from metrics.few_shots import eval_few_shots

    output = root / "origin"
    output.mkdir(parents=True, exist_ok=True)
    for folder, task in [("mmlu.json", "mmlu"), ("wmdp.json", "wmdp_bio")]:
        eval_few_shots(
            model_name=model, task_list=[task], output_path=str(output / folder)
        )
    scores = evaluation_scores(output, Path(model))
    result = {
        "model": model,
        "metrics": {task: accuracy for task, (_, accuracy) in scores.items()},
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


def collect(root):
    records = json.loads((root / "experiments.json").read_text())
    results = []
    for record in records:
        path = root / "metrics" / record["key"] / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    selected = {}
    for run in results:
        path = root / "metrics" / run["key"] / "runtime.json"
        if run["kind"] == "attack" and path.exists():
            runtime = json.loads(path.read_text())
            selected[(run["method"], run["samples"], run["epochs"], run["seed"])] = (
                runtime["selected_indices"]
            )
    methods = sorted({r["method"] for r in records if r["kind"] == "attack"})
    reference = methods[0] if methods else None
    cases = {
        (r["samples"], r["epochs"], r["seed"])
        for r in records
        if r["kind"] == "attack" and r["method"] == reference
    }
    expected_pairs = len(cases) * max(0, len(methods) - 1)
    paired = 0
    mismatches = []
    for method in methods[1:]:
        for samples, epochs, seed in sorted(cases):
            first = (reference, samples, epochs, seed)
            other = (method, samples, epochs, seed)
            if first in selected and other in selected:
                if selected[first] == selected[other]:
                    paired += 1
                else:
                    mismatches.append(
                        {
                            "method": method,
                            "samples": samples,
                            "epochs": epochs,
                            "seed": seed,
                        }
                    )
    epoch_groups = {
        (r["method"], r["seed"])
        for r in records
        if r["kind"] == "attack" and r["samples"] == 20
    }
    epoch_sets = 0
    epoch_mismatches = []
    for method, seed in sorted(epoch_groups):
        keys = [(method, 20, epoch, seed) for epoch in (1, 2, 3)]
        if all(key in selected for key in keys):
            if all(selected[key] == selected[keys[0]] for key in keys):
                epoch_sets += 1
            else:
                epoch_mismatches.append({"method": method, "seed": seed})
    summary = {
        "epoch_subsets_verified": epoch_sets,
        "epoch_subset_mismatches": epoch_mismatches,
        "pair_reference": reference,
        "paired_trials_expected": expected_pairs,
        "paired_trials_verified": paired,
        "paired_index_mismatches": mismatches,
        "expected": len(records),
        "completed": len(results),
        "runs": results,
        "complete": len(results) == len(records)
        and paired == expected_pairs
        and not mismatches
        and epoch_sets == len(epoch_groups)
        and not epoch_mismatches
        and (root / "origin/result.json").exists(),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot(root, output, labels=None, metric="UE"):
    import os
    import pandas as pd
    import seaborn as sns
    from matplotlib.lines import Line2D

    if metric not in {"UE", "MMLU"}:
        raise ValueError("Plot metric must be UE or MMLU")
    summary = collect(root)
    assert summary["complete"], "Incomplete experiments or original-model evaluation"
    names = {"npo": "NPO", "npo-sam": "NPO+SAM", **(labels or {})}
    methods = {r["method"] for r in summary["runs"]}
    if len({names.get(method, method) for method in methods}) != len(methods):
        raise ValueError("Use distinct display labels for distinct methods")
    rows = []
    for run in summary["runs"]:
        if run["kind"] != "attack":
            continue
        for sweep, axis in [
            ("Epochs (20 examples)", "epochs"),
            ("Examples (1 epoch)", "samples"),
        ]:
            if (axis == "epochs" and run["samples"] == 20) or (
                axis == "samples" and run["epochs"] == 1
            ):
                rows.append(
                    {
                        "Method": names.get(run["method"], run["method"]),
                        "MethodKey": run["method"],
                        "Sweep": sweep,
                        "Budget": run[axis],
                        "Trial": run["seed"],
                        "UE": 1 - run["metrics"]["wmdp_bio"],
                        "MMLU": run["metrics"]["mmlu"],
                    }
                )
    for run in summary["runs"]:
        if run["kind"] == "defense":
            for sweep in ["Epochs (20 examples)", "Examples (1 epoch)"]:
                rows.append(
                    {
                        "Method": names.get(run["method"], run["method"]),
                        "MethodKey": run["method"],
                        "Sweep": sweep,
                        "Budget": 0,
                        "Trial": 0,
                        "UE": 1 - run["metrics"]["wmdp_bio"],
                        "MMLU": run["metrics"]["mmlu"],
                    }
                )
    data = pd.DataFrame(rows)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    grid = sns.relplot(
        data=data,
        x="Budget",
        y=metric,
        hue="Method",
        col="Sweep",
        kind="line",
        marker="o",
        errorbar="se",
        facet_kws={"sharex": False},
        height=3.4,
    )
    reference = json.loads((root / "origin/result.json").read_text())
    for title, axis in grid.axes_dict.items():
        axis.set_xlabel(
            "Relearning epochs" if title.startswith("Epochs") else "Forget examples"
        )
        axis.set_xticks([0, 1, 2, 3] if title.startswith("Epochs") else [0, 20, 40, 60])
        axis.axhline(
            (
                1 - reference["metrics"]["wmdp_bio"]
                if metric == "UE"
                else reference["metrics"]["mmlu"]
            ),
            color="black",
            linestyle="--",
            label="Origin",
        )
    legend = dict(
        zip([t.get_text() for t in grid.legend.texts], grid.legend.legend_handles)
    )
    legend["Origin"] = Line2D([], [], color="black", linestyle="--")
    grid.legend.remove()
    grid.add_legend(legend_data=legend)
    grid.set_ylabels(
        "Unlearning effectiveness (1 − accuracy)"
        if metric == "UE"
        else "Full MMLU accuracy"
    )
    os.makedirs(os.path.dirname(str(output)) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")
    data.to_csv(os.path.splitext(str(output))[0] + ".csv", index=False)


def evaluate(root, record):
    """Run the authors' full task evaluations after the training process exits."""
    import subprocess

    config = json.loads(Path(record["config_path"]).read_text())
    output = Path(config["logger"]["json"]["root"]) / config["logger"]["name"]
    for folder, task in [
        ("mmlu.json", "mmlu"),
        ("wmdp.json", "wmdp" if record["kind"] == "attack" else "wmdp_bio"),
    ]:
        subprocess.run(
            [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={output / 'checkpoints'},cache_dir=./.cache,device_map=auto,parallelize=True",
                "--tasks",
                task,
                "--batch_size",
                "auto",
                "--output_path",
                str(output / folder),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "evaluate", "finish", "origin", "collect", "plot"],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/reproduce/figure3-npo-sam.json")
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metric", choices=["UE", "MMLU"], default="UE")
    args = parser.parse_args()
    if args.command == "prepare":
        records = prepare(
            args.source.resolve(),
            args.root.resolve(),
            args.cache.resolve(),
            json.loads(args.recipe.read_text()),
        )
        print(f"Prepared {len(records)} experiments in {args.root}")
    elif args.command in {"finish", "evaluate"}:
        records = json.loads((args.root / "experiments.json").read_text())
        record = next(r for r in records if r["key"] == args.key)
        result = (
            finish(args.root, record)
            if args.command == "finish"
            else evaluate(args.root, record)
        )
        print(json.dumps(result, indent=2))
    elif args.command == "origin":
        print(json.dumps(origin(args.root, args.model), indent=2))
    elif args.command == "collect":
        print(json.dumps(collect(args.root), indent=2))
    else:
        plot(args.root, args.output, metric=args.metric)
