import os
import shutil
import pytest
from PIL import Image
from unittest import mock
from src import preprocess


def create_dummy_image(path):
    img = Image.new("RGB", (100, 100), color="red")
    img.save(path)


def test_preprocess_creates_output(tmp_path, monkeypatch):
    # Create fake input structure
    input_dir = tmp_path / "petImages"
    cat_dir = input_dir / "cat"
    dog_dir = input_dir / "dog"
    cat_dir.mkdir(parents=True)
    dog_dir.mkdir(parents=True)

    create_dummy_image(cat_dir / "cat1.jpg")
    create_dummy_image(dog_dir / "dog1.jpg")

    output_dir = tmp_path / "processed"

    # Patch params
    monkeypatch.setattr(preprocess, "INPUT_DIR", str(input_dir))
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(preprocess, "IMAGE_SIZE", 64)
    monkeypatch.setattr(preprocess, "MAX_IMAGES", 1)

    # Mock MLflow
    with mock.patch("mlflow.start_run"), \
         mock.patch("mlflow.log_params"), \
         mock.patch("mlflow.log_metric"):

        preprocess.preprocess_images()

    assert (output_dir / "cats").exists()
    assert (output_dir / "dogs").exists()
