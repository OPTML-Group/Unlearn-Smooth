#!/bin/bash
#SBATCH --account=aip-grabus
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail
root=${1:?missing frozen run root}
kind=${2:?missing defense, attack, origin, or plot}
method=${3:-}
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface-datasets-unlearn-smooth"
export OMP_NUM_THREADS=8 MPLBACKEND=Agg
source "$SCRATCH/lilpipe/envs/unlearn-smooth-wmdp/bin/activate"
cd "$root/source/WMDP"
controller=scripts/misc/figure3.py
if [[ "$kind" == origin ]]; then
    python "$controller" origin --root "$root" --model "$method"
    exit
fi
if [[ "$kind" == plot ]]; then
    python "$controller" plot --root "$root" --output "$root/figures/figure3-npo-sam.pdf"
    python "$controller" plot --root "$root" --output "$root/figures/figure3-npo-sam.png"
    python "$controller" plot --root "$root" --metric MMLU --output "$root/figures/figure3-npo-sam-mmlu.pdf"
    exit
fi
mapfile -t fields < <(python - "$root" "$kind" "$method" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import json, sys
from pathlib import Path
root, kind, method, index = sys.argv[1:]
rows = [r for r in json.loads((Path(root) / 'experiments.json').read_text()) if r['kind'] == kind and r['method'] == method]
record = rows[int(index)]
print(record['key'])
print(record['config_path'])
print(record['expected_steps'])
PY
)
key=${fields[0]}
config=${fields[1]}
steps=${fields[2]}
metrics="$root/metrics/$key"
mkdir "$metrics"
if [[ "$kind" == attack ]]; then
    [[ ! -e "$root/attacks/$key" ]]
    entry=relearn
else
    [[ ! -e "$root/models/$key" ]]
    entry=unlearn
fi
python scripts/misc/figure3_runtime.py --entry "$entry" --config "$config" --report "$metrics/runtime.json" --expected-steps "$steps"
python "$controller" evaluate --root "$root" --key "$key"
python "$controller" finish --root "$root" --key "$key"
