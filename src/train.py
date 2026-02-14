import os
import yaml
import logging
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================================
# Logging Configuration
# =====================================================
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# Load DVC Params
# =====================================================
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_params = params["train"]

BASE_DIR = os.getcwd()
DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = train_params["batch_size"]
EPOCHS = train_params["epochs"]
LR = train_params["learning_rate"]
IMAGE_SIZE = train_params["image_size"]
VAL_SPLIT = train_params["validation_split"]

with open("experiments/experiment.yaml") as f:
    exp_cfg = yaml.safe_load(f)
experiment_name = exp_cfg["experiment_name"]
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment(experiment_name)

def train():

    with mlflow.start_run(run_name="SGD_Epoch_Training"):

        mlflow.log_params(train_params)

        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=VAL_SPLIT
        )

        train_gen = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="binary",
            subset="training"
        )

        val_gen = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="binary",
            subset="validation"
        )

        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        logger.info("Training started...")

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS
        )

        for epoch in range(EPOCHS):
            mlflow.log_metric("accuracy", history.history["accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)
            mlflow.log_metric("loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)

        model_path = os.path.join(MODEL_DIR, "model.h5")
        model.save(model_path)

        mlflow.tensorflow.log_model(model, "model")
        mlflow.log_artifact(model_path)

        logger.info("Training completed successfully.")


if __name__ == "__main__":
    train()
