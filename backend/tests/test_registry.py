"""Lightweight tests that run without the heavy ML deps (torch/terratorch/GDAL).

Full model inference isn't CI-tested — it needs GPU-class deps and ~GB model
downloads. These cover the pure-Python registry that wires tasks -> models, which
is exactly the surface most likely to drift when adding a task or model.
"""
from backend.models.registry import TASKS, list_models, resolve_model, task_source

EXPECTED_TASKS = {
    "buildings", "roads", "landcover", "embeddings", "burnscar", "flood", "textprompt",
}


def test_expected_tasks_present():
    assert set(TASKS) == EXPECTED_TASKS


def test_every_task_has_at_least_one_model():
    for task in TASKS:
        assert list_models(task), f"task {task!r} has no models"


def test_task_imagery_sources():
    assert task_source("buildings") == "naip"
    assert task_source("roads") == "naip"
    assert task_source("landcover") == "sentinel2"
    assert task_source("embeddings") == "sentinel2"
    assert task_source("flood") == "sentinel2"


def test_resolve_model_roundtrips_every_listed_model():
    for task in TASKS:
        for model in list_models(task):
            assert resolve_model(task, model["id"])["id"] == model["id"]
