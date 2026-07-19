import json

import torch
from torch import nn

from llm_lab.experiment import RunRecord, count_parameters, save_record, summarize


def test_parameter_counts_distinguish_frozen_parameters():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)
    counts = count_parameters(model)
    assert counts == {"total_parameters": 26, "trainable_parameters": 14}


def test_record_round_trip_and_summary(tmp_path):
    records = [
        RunRecord("ablation", 1, {"heads": 2}, {"loss": 2.0}, 0.1),
        RunRecord("ablation", 2, {"heads": 2}, {"loss": 4.0}, 0.2),
    ]
    path = tmp_path / "run.json"
    save_record(records[0], path)
    assert json.loads(path.read_text())["seed"] == 1
    assert summarize(records, "loss")["mean"] == 3.0


def test_seeded_torch_is_reproducible():
    from llm_lab.experiment import seed_everything

    seed_everything(42)
    first = torch.randn(5)
    seed_everything(42)
    torch.testing.assert_close(first, torch.randn(5))

