import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "figure3", ROOT / "scripts/misc/figure3.py"
)
figure3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure3)


def test_prepare_preserves_frozen_manifest(tmp_path):
    root = tmp_path / "run"
    recipe = json.loads((ROOT / "configs/reproduce/figure3-npo-sam.json").read_text())
    records = figure3.prepare(ROOT, root, tmp_path / "cache", recipe)
    manifest = (root / "experiments.json").read_bytes()
    configs = {p.name: p.read_bytes() for p in (root / "configs").iterdir()}
    recipe["defense_lr"] *= 2
    with pytest.raises(FileExistsError):
        figure3.prepare(ROOT, root, tmp_path / "cache", recipe)
    assert (root / "experiments.json").read_bytes() == manifest
    assert {p.name: p.read_bytes() for p in (root / "configs").iterdir()} == configs
    assert len(records) == 12
    for record in records:
        config = json.loads(Path(record["config_path"]).read_text())
        assert (
            config["dataset"]["batch_size"]
            * config["unlearn"]["gradient_accumulation_steps"]
            == 4
        )
