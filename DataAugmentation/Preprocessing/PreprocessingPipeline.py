import tensorflow as tf
import numpy as np

def create_preprocessing_pipeline(source_path):
    # Load dataset without batching
    ds = tf.keras.utils.image_dataset_from_directory(
        source_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=None,  # No batching
        shuffle=True
    )

    # Convert to grayscale
    def to_grayscale(images, labels):
        gray = tf.image.rgb_to_grayscale(images)  # (H, W, 1)
        return gray, labels

    ds = ds.map(to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)

    # Resize images to 67px wide and 50px high
    def resize_images(images, labels):
        resized = tf.image.resize(images, [50, 67])  # Resize to 50 (height) x 67 (width)
        return resized, labels

    ds = ds.map(resize_images, num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize images to [0, 1]
    def min_max_scale(images, labels):
        if images.dtype == tf.float32:
            images = tf.clip_by_value(images / tf.reduce_max(images), 0, 1)
        else:
            images = tf.image.convert_image_dtype(images, tf.float32)
        return images, labels

    ds = ds.map(min_max_scale, num_parallel_calls=tf.data.AUTOTUNE)

    # Adjust labels to start from 1
    def adjust_labels(images, labels):
        # Ensure labels are one-hot encoded
        labels = tf.reshape(labels, [-1])  # Ensure proper shape before argmax
        labels = tf.argmax(labels, axis=0) + 1
        return images, labels

    ds = ds.map(adjust_labels, num_parallel_calls=tf.data.AUTOTUNE)

    images_list = []
    labels_list = []

    for image, label in ds:
        images_list.append(image.numpy())
        labels_list.append(label.numpy())

    x = np.array(images_list)
    y = np.array(labels_list)

    return x, y

# Usage:
# X, y = create_preprocessing_pipeline("/path/to/dataset")
# model.fit(X, y, ...)


# import tensorflow as tf
#
# def create_preprocessing_pipeline(source_path, batch_size=5):
#     # Load dataset
#     ds = tf.keras.utils.image_dataset_from_directory(
#         source_path,
#         labels='inferred',
#         label_mode='categorical',
#         batch_size=batch_size,
#         shuffle=True
#     )
#
#     # Complexity reducing step 1: Convert to grayscale
#     def to_grayscale(images, labels):
#         gray = tf.image.rgb_to_grayscale(images)  # always do this
#         return gray, labels
#     ds = ds.map(to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)
#
#     # # Complexity reducing step 2: Min-max scaling (normalize to [0, 1])
#     # def min_max_scale(images, labels):
#     #     # Assuming images are in uint8 [0,255], convert to float32 [0,1]
#     #     images = tf.image.convert_image_dtype(images, tf.float32)
#     #     return images, labels
#
#     # ds = ds.map(min_max_scale, num_parallel_calls=tf.data.AUTOTUNE)
#
#     # Batch and prefetch
#     ds = ds.batch(batch_size, drop_remainder=True)
#     ds = ds.prefetch(tf.data.AUTOTUNE)
#
#     return ds
