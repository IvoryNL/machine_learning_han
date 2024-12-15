import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PreprocessingPipeline import create_preprocessing_pipeline
from tensorflow.keras import layers, models


def save_images_to_disk(x, y, source_path, output_folder_name="Augmented"):
    """
    Save images and their labels to disk.
    If y is one-hot encoded, it will be converted to integer labels.
    """

    augmented_path = os.path.join(source_path, output_folder_name)
    os.makedirs(augmented_path, exist_ok=True)

    # Convert one-hot to integer labels if needed
    if y.ndim > 1 and y.shape[1] > 1:
        labels = np.argmax(y, axis=1)
    else:
        labels = y

    # Determine if images are float in [0,1] or already uint8
    # If your pipeline returns float32 [0,1], we can directly save with scale=True.
    # If you prefer uint8, uncomment the conversion line below.
    # X = tf.image.convert_image_dtype(X, dtype=tf.uint8).numpy()

    image_counter = {}
    for i, (img, label) in enumerate(zip(x, labels)):
        label_folder = os.path.join(augmented_path, str(label))
        os.makedirs(label_folder, exist_ok=True)

        if label not in image_counter:
            image_counter[label] = 0

        file_name = f"img_{label}_{image_counter[label]}.jpg"
        file_path = os.path.join(label_folder, file_name)

        tf.keras.preprocessing.image.save_img(file_path, img, scale=True)

        image_counter[label] += 1

    print(f"Images have been saved to: {augmented_path}")

if __name__ == "__main__":
    source_path = r'D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset Ivan V3'
    batch_size = 5
    num_augmentations_per_image = 50
    epochs = 5

    # Create training dataset with augmentations
    x, y = create_preprocessing_pipeline(source_path, True, num_augmentations_per_image)

    # One-hot encode labels
    num_classes = 6  # Assuming there are 5 classes
    y = to_categorical(y, num_classes=num_classes)

    # save_images_to_disk(x, y, source_path=source_path, output_folder_name="Augmented_Results")

    # Convert x and y into a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(x), seed=42)

    # Calculate split sizes
    dataset_size = len(x)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    # Split into training and test datasets
    train_ds = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build the model with input shape (50, 67, 1)
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(50, 67, 1)),
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(6, activation='softmax')  # assuming 5 classes
    # ])

    # model = tf.keras.Sequential([
    #     layers.Input(shape=(50, 67, 1)),
    #     layers.Conv2D(32, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(128, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(6, activation='softmax')  # Adjust classes if needed
    # ])

    model = tf.keras.Sequential([
        layers.Input(shape=(50, 67, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Add dropout to prevent overfitting
        layers.Dense(6, activation='softmax')  # Adjust for the number of classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, epochs=epochs, validation_data=test_ds)