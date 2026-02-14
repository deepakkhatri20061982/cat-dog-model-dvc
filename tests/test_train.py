import pytest
from unittest import mock
from src import train

@mock.patch("mlflow.start_run")
@mock.patch("mlflow.log_params")
@mock.patch("mlflow.log_metric")
@mock.patch("mlflow.tensorflow.log_model")
def test_train_runs(
    mock_log_model,
    mock_log_metric,
    mock_log_params,
    mock_start_run
):
    with mock.patch("tensorflow.keras.preprocessing.image.ImageDataGenerator") as mock_gen:

        mock_train_gen = mock.Mock()
        mock_train_gen.samples = 10

        mock_val_gen = mock.Mock()
        mock_val_gen.samples = 5

        instance = mock_gen.return_value
        instance.flow_from_directory.side_effect = [mock_train_gen, mock_val_gen]

        with mock.patch("tensorflow.keras.Sequential") as mock_model:
            model_instance = mock_model.return_value

            # 👇 FIX: match number of epochs
            epochs = train.EPOCHS

            model_instance.fit.return_value.history = {
                "accuracy": [0.8] * epochs,
                "val_accuracy": [0.75] * epochs,
                "loss": [0.5] * epochs,
                "val_loss": [0.6] * epochs
            }

            train.train()

    assert mock_log_params.called
