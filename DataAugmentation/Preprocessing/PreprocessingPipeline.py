import tensorflow as tf
import numpy as np
from DataAugmentationLayer import DataAugmentationLayer

def create_preprocessing_pipeline(source_path, is_training, num_augmentations):
    # Load dataset without batching, use categorical labels
    ds = tf.keras.utils.image_dataset_from_directory(
        source_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=None,  # No batching, for custom augmentation flow
        shuffle=True
    )

    # Convert to grayscale
    def to_grayscale(images, labels):
        gray = tf.image.rgb_to_grayscale(images)  # (H, W, 1)
        return gray, labels

    ds = ds.map(to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)

    # Resize images to fixed size: 50 (H) x 67 (W)
    def resize_images(images, labels):
        resized = tf.image.resize(images, [50, 67])
        return resized, labels

    ds = ds.map(resize_images, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        augment_layer = DataAugmentationLayer()

        def augment_images(image, label):
            # Add a batch dimension of size num_augmentations
            image = tf.expand_dims(image, axis=0)
            image = tf.repeat(image, repeats=[num_augmentations], axis=0)

            label = tf.expand_dims(label, 0)
            label = tf.repeat(label, repeats=[num_augmentations], axis=0)
            label = tf.squeeze(label)

            augmented = augment_layer(image, training=True)
            return augmented, label

        ds = ds.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.unbatch()

    # Normalize images to [0, 1]
    def min_max_scale(images, labels):
        max_val = tf.reduce_max(images)
        images = tf.cond(
            tf.equal(max_val, 0),
            lambda: images,
            lambda: tf.clip_by_value(images / max_val, 0, 1)
        )
        return images, labels

    ds = ds.map(min_max_scale, num_parallel_calls=tf.data.AUTOTUNE)

    images_list = []
    labels_list = []

    for image, label in ds:
        images_list.append(image.numpy())
        labels_list.append(label.numpy())

    x = np.array(images_list)
    y = np.array(labels_list)

    # Convert one-hot labels (0-based) to integer labels (1-based)
    # If y is shape (N, 5), argmax along axis=1 gives values in [0..4],
    # Adding 1 makes them [1..5]
    y_integers = np.argmax(y, axis=1) + 1

    return x, y_integers


# import tensorflow as tf
# import numpy as np
# from DataAugmentationLayer import DataAugmentationLayer
#
# def create_preprocessing_pipeline(source_path, is_training, num_augmentations):
#     # Load dataset without batching
#     ds = tf.keras.utils.image_dataset_from_directory(
#         source_path,
#         labels='inferred',
#         label_mode='categorical',
#         batch_size=None,  # No batching
#         shuffle=True
#     )
#
#     # Convert to grayscale
#     def to_grayscale(images, labels):
#         gray = tf.image.rgb_to_grayscale(images)  # (H, W, 1)
#         return gray, labels
#
#     ds = ds.map(to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)
#
#     # Resize images to 67px wide and 50px high
#     def resize_images(images, labels):
#         resized = tf.image.resize(images, [50, 67])  # Resize to 50 (height) x 67 (width)
#         return resized, labels
#
#     ds = ds.map(resize_images, num_parallel_calls=tf.data.AUTOTUNE)
#
#     # If training, augment
#     if is_training:
#         augment_layer = DataAugmentationLayer()
#
#         def augment_images(image, label):
#             image = tf.expand_dims(image, axis=0)  # shape (1,h,w,1)
#             # Repeat image for num_augmentations_per_image
#             image = tf.repeat(image, repeats=[num_augmentations], axis=0)
#             # Repeat labels but collapse into single labels per augmented image
#             label = tf.expand_dims(label, 0)  # Add a batch dimension
#             label = tf.repeat(label, repeats=[num_augmentations], axis=0)  # Duplicate for augmentations
#             label = tf.squeeze(label)  # Remove unnecessary dimensions
#             augmented = augment_layer(image, training=True)
#             return augmented, label
#
#         ds = ds.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
#         ds = ds.unbatch()
#
#     # Normalize images to [0, 1]
#     def min_max_scale(images, labels):
#         if images.dtype == tf.float32:
#             images = tf.clip_by_value(images / tf.reduce_max(images), 0, 1)
#         else:
#             images = tf.image.convert_image_dtype(images, tf.float32)
#         return images, labels
#
#     ds = ds.map(min_max_scale, num_parallel_calls=tf.data.AUTOTUNE)
#
#     # # Adjust labels to start from 1
#     # def adjust_labels(images, labels):
#     #     # Ensure labels are one-hot encoded
#     #     labels = tf.reshape(labels, [-1])  # Ensure proper shape before argmax
#     #     labels = tf.argmax(labels, axis=0) + 1
#     #     return images, labels
#     #
#     # ds = ds.map(adjust_labels, num_parallel_calls=tf.data.AUTOTUNE)
#
#     images_list = []
#     labels_list = []
#
#     for image, label in ds:
#         images_list.append(image.numpy())
#         labels_list.append(label.numpy())
#
#     x = np.array(images_list)
#     y = np.array(labels_list)
#
#     # print(y)
#
#     return x, y
