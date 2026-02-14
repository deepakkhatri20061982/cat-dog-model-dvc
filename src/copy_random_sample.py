import os
import random
import shutil
import logging

# =====================================================
# Configuration
# =====================================================
SOURCE_DIR = "data/petImagesRaw"   # Adjust if needed
DEST_DIR = "data/petImages"
NUM_IMAGES = 2000
SEED = 42  # For reproducibility

# =====================================================
# Logging Setup
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

random.seed(SEED)


def copy_random_images(class_name):
    src_class_dir = os.path.join(SOURCE_DIR, class_name)
    dest_class_dir = os.path.join(DEST_DIR, class_name)

    os.makedirs(dest_class_dir, exist_ok=True)

    if not os.path.exists(src_class_dir):
        raise FileNotFoundError(f"Source folder not found: {src_class_dir}")

    all_images = [
        f for f in os.listdir(src_class_dir)
        if os.path.isfile(os.path.join(src_class_dir, f))
    ]

    logger.info(f"{class_name}: Total images available: {len(all_images)}")

    if len(all_images) < NUM_IMAGES:
        raise ValueError(f"Not enough images in {class_name} folder")

    selected_images = random.sample(all_images, NUM_IMAGES)

    for img_name in selected_images:
        shutil.copy2(
            os.path.join(src_class_dir, img_name),
            os.path.join(dest_class_dir, img_name)
        )

    logger.info(f"{class_name}: {NUM_IMAGES} images copied successfully.")


def main():
    logger.info("Starting subset creation...")

    for class_name in ["Cat", "Dog"]:
        copy_random_images(class_name)

    logger.info("Subset creation completed successfully.")


if __name__ == "__main__":
    main()
