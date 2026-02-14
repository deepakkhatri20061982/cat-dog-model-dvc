import os
import logging
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
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
        logging.FileHandler("training.log")
    ]
)

logger = logging.getLogger(__name__)

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
PREDS_DIR = os.path.join(OUTPUT_DIR, "preds")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

USE_MLFLOW = True


with open("experiments/experiment.yaml") as f:
    exp_cfg = yaml.safe_load(f)
experiment_name = exp_cfg["experiment_name"]
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment(experiment_name)

# =====================================================
# Custom Callback for Logging + MLflow
# =====================================================
class MLflowLoggingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Epoch {epoch + 1} started")

    def on_epoch_end(self, epoch, logs=None):
        logger.info(
            f"Epoch {epoch + 1} completed | "
            f"Loss: {logs['loss']:.4f} | "
            f"Accuracy: {logs['accuracy']:.4f} | "
            f"Val_Loss: {logs['val_loss']:.4f} | "
            f"Val_Accuracy: {logs['val_accuracy']:.4f}"
        )

        if USE_MLFLOW:
            mlflow.log_metrics({
                "loss": logs["loss"],
                "accuracy": logs["accuracy"],
                "val_loss": logs["val_loss"],
                "val_accuracy": logs["val_accuracy"]
            }, step=epoch)


# =====================================================
# Data Generators
# =====================================================
logger.info("Preparing data generators...")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

logger.info(f"Training samples: {train_gen.samples}")
logger.info(f"Validation samples: {val_gen.samples}")

# =====================================================
# Save Training Samples
# =====================================================
images, labels = next(train_gen)
for i in range(8):
    plt.imshow(images[i])
    plt.title("Dog" if labels[i] == 1 else "Cat")
    plt.axis("off")
    plt.savefig(os.path.join(SAMPLES_DIR, f"train_sample_{i}.png"))
    plt.clf()

logger.info("Saved training sample images.")

# =====================================================
# Model Definition
# =====================================================
logger.info("Building model...")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

logger.info("Model compiled successfully.")

# =====================================================
# MLflow Setup
# =====================================================
if USE_MLFLOW:
    mlflow.start_run()

    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("image_size", 224)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "binary_crossentropy")
    mlflow.log_param("train_samples", train_gen.samples)
    mlflow.log_param("val_samples", val_gen.samples)

    mlflow.set_tag("project", "Cats_vs_Dogs")
    mlflow.set_tag("framework", "TensorFlow")

# =====================================================
# Training
# =====================================================
logger.info("Starting training...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3,
    callbacks=[MLflowLoggingCallback()]
)

logger.info("Training completed.")

# =====================================================
# Save Model
# =====================================================
model_path = os.path.join(MODEL_DIR, "model.h5")
model.save(model_path)
logger.info(f"Model saved at {model_path}")

if USE_MLFLOW:
    mlflow.tensorflow.log_model(model, "model")
    mlflow.log_artifact(model_path)
    mlflow.log_artifacts(SAMPLES_DIR, artifact_path="train_samples")

# =====================================================
# Save Prediction Samples
# =====================================================
logger.info("Generating prediction samples...")

val_gen.reset()
images, labels = next(val_gen)
preds = model.predict(images)

for i in range(8):
    pred_label = "Dog" if preds[i][0] > 0.5 else "Cat"
    true_label = "Dog" if labels[i] == 1 else "Cat"

    plt.imshow(images[i])
    plt.title(f"Pred: {pred_label} | True: {true_label}")
    plt.axis("off")
    plt.savefig(os.path.join(PREDS_DIR, f"pred_{i}.png"))
    plt.clf()

logger.info("Saved prediction samples.")

if USE_MLFLOW:
    mlflow.log_artifacts(PREDS_DIR, artifact_path="predictions")
    mlflow.end_run()

logger.info("Pipeline execution finished successfully.")
