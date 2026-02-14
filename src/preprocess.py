import os
import yaml
import logging
import mlflow
from PIL import Image

# =====================================================
# Logging Configuration
# =====================================================
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocess.log")
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# Load DVC Params
# =====================================================
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

preprocess_params = params["preprocess"]

BASE_DIR = os.getcwd()
INPUT_DIR = preprocess_params["input_dir"]
OUTPUT_DIR = preprocess_params["output_dir"]
IMAGE_SIZE = preprocess_params["image_size"]
MAX_IMAGES = preprocess_params["max_images"]

with open("experiments/experiment.yaml") as f:
    exp_cfg = yaml.safe_load(f)
experiment_name = exp_cfg["experiment_name"]
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment(experiment_name)

def preprocess_images():

    with mlflow.start_run(run_name="preprocess"):

        # Log DVC params to MLflow
        mlflow.log_params(preprocess_params)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        mapping = {"cats": "cat", "dogs": "dog"}

        total_processed = 0
        total_skipped = 0

        for out_label, in_label in mapping.items():

            in_path = os.path.join(INPUT_DIR, in_label)
            out_path = os.path.join(OUTPUT_DIR, out_label)
            os.makedirs(out_path, exist_ok=True)

            logger.info(f"Processing class: {out_label}")

            count = 0
            skipped = 0

            for img_name in os.listdir(in_path):
                img_path = os.path.join(in_path, img_name)

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                        img.save(os.path.join(out_path, img_name))
                        count += 1

                    if count >= MAX_IMAGES:
                        break

                except Exception as e:
                    skipped += 1
                    logger.warning(f"Skipping {img_name}: {e}")

            logger.info(f"{out_label} processed: {count}")
            logger.info(f"{out_label} skipped: {skipped}")

            mlflow.log_metric(f"{out_label}_processed", count)
            mlflow.log_metric(f"{out_label}_skipped", skipped)

            total_processed += count
            total_skipped += skipped

        mlflow.log_metric("total_processed", total_processed)
        mlflow.log_metric("total_skipped", total_skipped)

        logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_images()
