import os
import logging
from PIL import Image
import mlflow
import mlflow.pytorch
import yaml

# ==============================
# Resolve Project Base Directory
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# Configure Logger
# ==============================
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def preprocess_images(
    input_dir=os.path.join(BASE_DIR, "data", "petImages"),
    output_dir=os.path.join(BASE_DIR, "data", "processed"),
    size=(224, 224),
    max_images=2000
):
    with open("experiments/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    experiment_name = exp_cfg["experiment_name"]
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    """
    Preprocess Cat and Dog images:
    - Resize images
    - Convert to RGB
    - Skip broken images
    - Log metadata to MLflow
    """

    mapping = {"cats": "cat", "dogs": "dog"}
    os.makedirs(output_dir, exist_ok=True)

    # ==============================
    # Start MLflow Run
    # ==============================
    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("image_width", size[0])
        mlflow.log_param("image_height", size[1])
        mlflow.log_param("max_images_per_class", max_images)
        mlflow.log_param("input_directory", input_dir)
        mlflow.log_param("output_directory", output_dir)

        mlflow.set_tag("project", "Cats_vs_Dogs")
        mlflow.set_tag("stage", "data_preprocessing")

        total_processed = 0
        total_skipped = 0

        for out_label, in_label in mapping.items():
            in_path = os.path.join(input_dir, in_label)
            out_path = os.path.join(output_dir, out_label)
            os.makedirs(out_path, exist_ok=True)

            if not os.path.exists(in_path):
                logger.error(f"Input path not found: {in_path}")
                raise FileNotFoundError(f"Input path not found: {in_path}")

            logger.info(f"Processing class: {out_label}")
            count = 0
            skipped = 0

            for img_name in os.listdir(in_path):
                img_path = os.path.join(in_path, img_name)

                if not os.path.isfile(img_path):
                    continue

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img = img.resize(size)
                        img.save(os.path.join(out_path, img_name))
                        count += 1

                    if count >= max_images:
                        break

                except Exception as e:
                    skipped += 1
                    logger.warning(f"Skipping {img_name}: {e}")
                    continue

            logger.info(f"{out_label}: {count} images processed")
            logger.info(f"{out_label}: {skipped} images skipped")

            # Log per-class metrics
            mlflow.log_metric(f"{out_label}_processed", count)
            mlflow.log_metric(f"{out_label}_skipped", skipped)

            total_processed += count
            total_skipped += skipped

        # Log overall metrics
        mlflow.log_metric("total_processed", total_processed)
        mlflow.log_metric("total_skipped", total_skipped)

        # Log processed dataset as artifact
        mlflow.log_artifacts(output_dir, artifact_path="processed_data")

        logger.info("Preprocessing completed successfully.")
        logger.info(f"Total Processed: {total_processed}")
        logger.info(f"Total Skipped: {total_skipped}")


if __name__ == "__main__":
    preprocess_images()
