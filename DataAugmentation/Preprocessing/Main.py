import os
import numpy as np
import tensorflow as tf
from sympy import false

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
    num_augmentations_per_image = 10
    epochs = 5

    # Create training dataset with augmentations
    x, y = create_preprocessing_pipeline(source_path, True, num_augmentations_per_image)

    save_images_to_disk(x, y, source_path=source_path, output_folder_name="Augmented_Results")

    # Save augmented data to disk
    # save_augmented_data(train_ds, source_path)

    # # Build and train the model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(224, 224, 1)),
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(5, activation='softmax')  # assuming 5 classes
    # ])
    #
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(train_ds, epochs=epochs)