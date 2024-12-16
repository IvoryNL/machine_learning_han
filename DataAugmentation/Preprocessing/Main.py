import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import keyboard
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.optimizers.schedules import CosineDecay
from PreprocessingPipeline import create_preprocessing_pipeline


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
    batch_size = 16
    num_augmentations_per_image = 50
    epochs = 10

    # Create training dataset with augmentations
    x, y = create_preprocessing_pipeline(source_path, True, num_augmentations_per_image)

    # One-hot encode labels
    num_classes = 6
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

    print(f"Batch size: {batch_size}, Epoch size: {epochs}")
    print(f"Train set size: {train_size}, Test set size: {test_size}")

    # Split into training and test datasets
    train_ds = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # model = tf.keras.Sequential([
    #     layers.Input(shape=(50, 67, 1)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(128, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(256, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(32, activation='relu'),
    #     layers.Dense(6, activation='softmax')  # Adjust for the number of classes
    # ])
    model = tf.keras.Sequential([
        layers.Input(shape=(50, 67, 1)),
        layers.Conv2D(32, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])
    # Set the learning rate scheduler (Cosine Decay)
    initial_learning_rate = 0.001
    lr_scheduler = CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=10 * len(train_ds),  # Total steps for full decay (epochs * batches per epoch)
        alpha=0.0001  # Final learning rate fraction of initial (10% of 0.001)
    )

    optimizer = optimizers.Adam(learning_rate=lr_scheduler)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(train_ds, epochs=epochs, validation_data=test_ds)

    # # Define the learning rate scheduler callback
    # lr_scheduler = callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',  # Monitor validation loss
    #     factor=0.5,  # Reduce learning rate by this factor
    #     patience=0,  # Number of epochs with no improvement to wait before reducing
    #     min_lr=1e-6,  # Minimum learning rate
    #     verbose=1  # Print learning rate updates
    # )

    # Add an early stopping callback to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model (without passing lr_scheduler as a callback)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=[early_stopping]  # Only valid callbacks here
    )

    # Train the model and store the history
    # history = model.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=[lr_scheduler])
    # history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # steps = range(10 * len(train_ds))  # Total training steps
    # lr_values = [lr_scheduler(step) for step in steps]
    # plt.plot(steps, lr_values)
    # plt.title('Learning Rate Schedule (Cosine Decay)')
    # plt.xlabel('Training Steps')
    # plt.ylabel('Learning Rate')
    # plt.grid(True)
    plt.show()

    # 1
    # model = tf.keras.Sequential([
    #     layers.Input(shape=(50, 67, 1)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(128, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(256, (3, 3), activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(32, activation='relu'),
    #     layers.Dense(6, activation='softmax')  # Adjust for the number of classes
    # ])