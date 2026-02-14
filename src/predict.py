import os
import random
import logging
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# =====================================================
# Logging Configuration
# =====================================================
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("prediction.log")
    ]
)

logger = logging.getLogger(__name__)

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "predictions_random")

os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_MLFLOW = True
NUM_SAMPLES = 15
IMAGE_SIZE = 224

with open("experiments/experiment.yaml") as f:
    exp_cfg = yaml.safe_load(f)
experiment_name = exp_cfg["experiment_name"]
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment(experiment_name)

# =====================================================
# Load Model
# =====================================================
logger.info("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")


# =====================================================
# Collect All Image Paths
# =====================================================
class_names = sorted(os.listdir(DATA_DIR))  # ['cats', 'dogs']
image_paths = []

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(class_dir):
        image_paths.append((os.path.join(class_dir, img_name), class_name))

logger.info(f"Total images found: {len(image_paths)}")

# Random sample selection
sample_images = random.sample(image_paths, NUM_SAMPLES)


# =====================================================
# MLflow Setup
# =====================================================
if USE_MLFLOW:
    mlflow.start_run()

    mlflow.log_param("num_samples", NUM_SAMPLES)
    mlflow.log_param("image_size", IMAGE_SIZE)


# =====================================================
# Prediction Loop
# =====================================================
correct = 0
results = []

logger.info("Starting predictions...")

for idx, (img_path, true_label) in enumerate(sample_images):

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred_prob = model.predict(img_array, verbose=0)[0][0]
    pred_label = "dogs" if pred_prob > 0.5 else "cats"

    if pred_label == true_label:
        correct += 1

    logger.info(
        f"Image: {os.path.basename(img_path)} | "
        f"True: {true_label} | Predicted: {pred_label} | "
        f"Confidence: {pred_prob:.4f}"
    )

    print(
        f"Image: {os.path.basename(img_path)} | "
        f"True: {true_label} | Predicted: {pred_label} | "
        f"Confidence: {pred_prob:.4f}"
    )

    # Save prediction image
    plt.imshow(img)
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis("off")
    save_path = os.path.join(OUTPUT_DIR, f"pred_{idx}.png")
    plt.savefig(save_path)
    plt.clf()

    results.append({
        "image": os.path.basename(img_path),
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence": float(pred_prob)
    })


# =====================================================
# Final Metrics
# =====================================================
accuracy = correct / NUM_SAMPLES
logger.info(f"Prediction Accuracy on random samples: {accuracy:.4f}")
print(f"\nPrediction Accuracy: {accuracy:.4f}")


# =====================================================
# MLflow Logging
# =====================================================
if USE_MLFLOW:

    mlflow.log_metric("random_sample_accuracy", accuracy)

    # Log prediction artifacts
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="random_predictions")

    # Log prediction table as JSON
    import json
    results_path = os.path.join(OUTPUT_DIR, "prediction_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    mlflow.log_artifact(results_path)

    mlflow.end_run()

logger.info("Prediction pipeline completed successfully.")
