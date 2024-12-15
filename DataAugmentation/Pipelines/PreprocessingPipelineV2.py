import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.layers import Rescaling

class PreprocessingPipeline:
    def __init__(self, image_size=(50, 67)):  # Height x Width
        self.image_size = image_size
        self.pipeline = tf.keras.Sequential([
            # tf.keras.layers.Resizing(image_size[0], image_size[1]),  # Resize to 50x67
            tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),  # Convert to grayscale
            tf.keras.layers.Lambda(self.random_rotation),  # Apply random rotation
            tf.keras.layers.Rescaling(1.0 / 255.0),                 # Normalize to [0, 1]
        ])

    def random_rotation(self, image):
        """
        Randomly rotate the image left or right by up to ±10 degrees.
        Fills the background with the mean value of the 4 corner pixels.
        """
        # Generate a random angle in radians between -10 and +10 degrees
        angle = tf.random.uniform([], minval=-15 / 360, maxval=15 / 360, dtype=tf.float32) * 2 * np.pi

        # Rotate the image without interpolation
        rotated_image = tfa.image.rotate(image, angles=angle)

        # Compute mean of corner pixels
        corners = tf.stack([
            rotated_image[0, 0],  # Top-left corner
            rotated_image[0, -1],  # Top-right corner
            rotated_image[-1, 0],  # Bottom-left corner
            rotated_image[-1, -1],  # Bottom-right corner
        ])
        mean_value = tf.reduce_mean(corners)

        # Create a mask for black pixels (background caused by rotation)
        mask = tf.cast(rotated_image == 0, tf.float32)  # 1 for black pixels, 0 otherwise

        # Replace black pixels with the mean corner value
        filled_image = rotated_image + mask * mean_value

        return filled_image

    def process_dataset(self, source_path, batch_size=32):
        """
        Load images from a directory and apply preprocessing. Ensure labels match folder names.

        Args:
            source_path (str): Path to the image directory.
            batch_size (int): Batch size for processing.

        Returns:
            tf.data.Dataset: Preprocessed dataset with tensors.
        """
        # Load dataset with folder names as class names
        dataset = tf.keras.utils.image_dataset_from_directory(
            source_path,
            # image_size=self.image_size,
            batch_size=batch_size,
            class_names=['1', '2', '3', '4', '5']  # Explicitly set folder names as class names
        )

        # Create a mapping from class indices to folder names
        class_names = tf.constant(dataset.class_names)  # TensorFlow constant
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.range(len(class_names)),  # Keys: indices (0, 1, 2, ...)
                values=tf.strings.to_number(class_names, tf.int32)  # Values: folder names as integers (1, 2, ...)
            ),
            default_value=-1  # Default value if key not found
        )

        # Map numeric labels to folder names
        dataset = dataset.map(lambda x, y: (self.pipeline(x), table.lookup(y)))

        return dataset

