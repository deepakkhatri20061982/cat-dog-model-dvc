from datetime import datetime
import yaml
import os

os.makedirs("experiments", exist_ok=True)

experiment_name = f"MNIST_DVC_LogisticRegressionV2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

data = {
    "experiment_name": experiment_name
}

with open("experiments/experiment.yaml", "w") as f:
    yaml.safe_dump(data, f)

print("Experiment created:", experiment_name)
