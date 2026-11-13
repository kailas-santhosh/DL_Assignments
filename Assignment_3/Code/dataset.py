"""
dataset.py

Dataset loading utilities for Assignment 3.
"""

import tensorflow as tf
import config


# =============================================================================
# Image Preprocessing
# =============================================================================

def preprocess(image, label):
    """
    Normalize image, flatten to 784 features,
    and one-hot encode labels.
    """

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(
        image,
        (config.INPUT_DIM,),
    )

    label = tf.one_hot(
        label,
        depth=config.NUM_CLASSES,
    )

    return image, label


# =============================================================================
# Dataset Loader
# =============================================================================

def load_dataset(directory, shuffle=True):
    """
    Load dataset from directory.

    Returned dataset is UNBATCHED.
    """

    dataset = tf.keras.utils.image_dataset_from_directory(

        directory,

        labels="inferred",

        label_mode="int",

        color_mode="grayscale",

        image_size=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        ),

        batch_size=None,

        shuffle=shuffle,

        seed=config.RANDOM_SEED,
    )

    dataset = dataset.map(

        preprocess,

        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:

        dataset = dataset.shuffle(

            2048,

            seed=config.RANDOM_SEED,

            reshuffle_each_iteration=True,
        )

    dataset = dataset.cache()

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# =============================================================================
# Count Images
# =============================================================================

def count_images(directory):

    total = 0

    for class_dir in directory.iterdir():

        if class_dir.is_dir():

            total += len(list(class_dir.glob("*")))

    return total


# =============================================================================
# Load All Datasets
# =============================================================================

def get_datasets():

    train_size = count_images(config.TRAIN_DIR)

    config.FULL_BATCH_SIZE = train_size

    train_dataset = load_dataset(

        config.TRAIN_DIR,

        shuffle=True,
    )

    validation_dataset = load_dataset(

        config.VAL_DIR,

        shuffle=False,
    )

    test_dataset = load_dataset(

        config.TEST_DIR,

        shuffle=False,
    )

    return (

        train_dataset,

        validation_dataset,

        test_dataset,

        train_size,
    )


# =============================================================================
# Batch Dataset
# =============================================================================

def get_dataset_with_batch(
    dataset,
    batch_size,
):
    """
    Batch dataset according to optimizer.
    """

    return dataset.batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


# =============================================================================
# Dataset Information
# =============================================================================

def print_dataset_info(train_size):

    validation_size = count_images(config.VAL_DIR)

    test_size = count_images(config.TEST_DIR)

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Classes          : {config.CLASS_NAMES}")
    print(f"Input Dimension  : {config.INPUT_DIM}")
    print(f"Training Images  : {train_size}")
    print(f"Validation Images: {validation_size}")
    print(f"Testing Images   : {test_size}")

    print("=" * 60)
