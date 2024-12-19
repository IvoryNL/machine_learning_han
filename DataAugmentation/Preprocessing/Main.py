import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, callbacks, models, optimizers
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.optimizers.schedules import CosineDecay
from PreprocessingPipeline import create_preprocessing_pipeline

class F1MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        self.f1_scores = []
        self.precisions = []
        self.recalls = []

    def on_epoch_end(self, epoch, logs=None):
        # Extract validation data
        x_val = []
        y_val_onehot = []
        for x, y in self.val_data.unbatch():
            x_val.append(x.numpy())
            y_val_onehot.append(y.numpy())

        x_val = np.array(x_val)
        y_val_onehot = np.array(y_val_onehot)

        # Get predictions
        y_pred_prob = self.model.predict(x_val)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_val_onehot, axis=1)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        self.f1_scores.append(f1)
        self.precisions.append(precision)
        self.recalls.append(recall)

        # print(f"Epoch {epoch+1} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    source_path = r'D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset Ivan V3'
    batch_size = 16
    num_augmentations_per_image = 50
    epochs = 25

    # Create training dataset with augmentations
    x, y = create_preprocessing_pipeline(source_path, True, num_augmentations_per_image)

    # Convert labels from [1..5] to [0..4] for training, then one-hot encode later
    y_zero_based = y - 1
    num_classes = 5

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y_zero_based))
    dataset = dataset.shuffle(buffer_size=len(x), seed=42)

    # Split dataset
    dataset_size = len(x)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    print(f"Train set size: {train_size}, Test set size: {test_size}")

    train_ds = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # One-hot encode labels in the dataset pipeline
    def one_hot_labels(images, labels):
        return images, tf.one_hot(labels, num_classes)

    train_ds = train_ds.map(one_hot_labels)
    test_ds = test_ds.map(one_hot_labels)

    model = tf.keras.Sequential([
        layers.Input(shape=(50, 67, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Learning rate scheduler
    initial_learning_rate = 0.001
    lr_scheduler = CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=epochs * len(train_ds),
        alpha=0.0001
    )

    optimizer = optimizers.Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
    )

    # Setup callback to track F1, Precision, Recall
    f1_callback = F1MetricsCallback(val_data=test_ds)

    # Train the model
    history = model.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=[early_stopping, f1_callback])

    # Plot Accuracy & Loss in one figure
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='blue', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(history.history['loss'], label='Train Loss', color='red')
    ax2.plot(history.history['val_loss'], label='Val Loss', color='red', linestyle='--')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    plt.title('Epoch vs Accuracy & Loss')
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(600, 500)
    manager.window.wm_geometry("+0+0")

    # Evaluate final predictions for confusion matrix and metrics
    x_test_list = []
    y_test_list = []
    for img, lbl in test_ds.unbatch():
        x_test_list.append(img.numpy())
        y_test_list.append(lbl.numpy())
    x_test_np = np.array(x_test_list)
    y_test_np = np.array(y_test_list)
    y_true = np.argmax(y_test_np, axis=1)  # 0-based

    y_pred_prob = model.predict(x_test_np)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize row-wise
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix with percentages
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2%',
                xticklabels=range(1, num_classes + 1),
                yticklabels=range(1, num_classes + 1))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix (Percentages)')
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(600, 500)
    manager.window.wm_geometry("+900+0")

    # Compute final F1, Precision, and Recall (weighted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Plot F1, Precision, and Recall as bars
    metrics = ['F1 Score', 'Precision', 'Recall']
    values = [f1, precision, recall]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['skyblue', 'green', 'orange'])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    plt.ylim(0, 1.1)
    plt.title('Final F1, Precision, and Recall')
    plt.ylabel('Metric Value')
    plt.grid(axis='y')
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(600, 500)
    manager.window.wm_geometry("+0+800")

    # Plot F1, Precision, and Recall in one figure
    # F1 as a bar chart, Precision and Recall as line charts
    epochs_range = range(1, len(f1_callback.f1_scores) + 1)
    plt.figure(figsize=(10,6))

    # Plot F1 as bar
    plt.bar(epochs_range, f1_callback.f1_scores, color='skyblue', label='F1 Score')

    # Plot Precision and Recall as lines
    plt.plot(epochs_range, f1_callback.precisions, color='green', marker='o', label='Precision')
    plt.plot(epochs_range, f1_callback.recalls, color='orange', marker='s', label='Recall')

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('F1, Precision, and Recall over Epochs')
    plt.xticks(epochs_range)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(600, 500)
    manager.window.wm_geometry("+900+800")

    plt.show()