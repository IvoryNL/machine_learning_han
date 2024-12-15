import tensorflow as tf
import tensorflow_addons as tfa
import math

class DataAugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # You can add any initialization parameters if needed.

    def call(self, images, training=None):
        # images expected shape: (batch, h, w, 1) after grayscale conversion
        if training:
            # Extract corner mean before transformations
            corner_mean_expanded = self.extract_corner_mean(images)
            images = self.random_rotate(images, corner_mean_expanded)

            corner_mean_expanded = self.extract_corner_mean(images)
            images = self.random_translate(images, corner_mean_expanded)

            images = self.random_brightness(images)
            images = self.random_focus(images)
            images = self.random_scale(images)
        return images

    def random_rotate(self, images, corner_mean_expanded):
        batch_size = tf.shape(images)[0]

        # 50% chance to pick angle from 0-15 degrees, else 345-360
        choose_low_range = tf.random.uniform([batch_size], 0, 1) > 0.5
        low_angles = tf.random.uniform([batch_size], 0, math.radians(15))
        high_angles = tf.random.uniform([batch_size], math.radians(345), math.radians(360))
        angles = tf.where(choose_low_range, low_angles, high_angles)

        rotated_images = tfa.image.rotate(images, angles, fill_mode='constant', fill_value=0.0)
        rotated_images_filled = self.fill_black_with_mean(rotated_images, corner_mean_expanded)
        return rotated_images_filled

    def random_translate(self, images, corner_mean_expanded):
        batch_size = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]

        # Translate between -15% and 15% of the dimensions
        tx = tf.random.uniform([batch_size], -0.15, 0.15) * tf.cast(h, tf.float32)
        ty = tf.random.uniform([batch_size], -0.15, 0.15) * tf.cast(w, tf.float32)

        translations = tf.stack([tx, ty], axis=1)
        translated_images = tfa.image.translate(images, translations, fill_mode='constant', fill_value=0.0)
        translated_images_filled = self.fill_black_with_mean(translated_images, corner_mean_expanded)
        return translated_images_filled

    def extract_corner_mean(self, images):
        # images: (batch, h, w, 1)
        batch_size = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]

        # Extract corners from the original images before any transform
        top_left = images[:, 0, 0, :]              # (batch, 1)
        top_right = images[:, 0, w - 1, :]         # (batch, 1)
        bottom_left = images[:, h - 1, 0, :]       # (batch, 1)
        bottom_right = images[:, h - 1, w - 1, :]  # (batch, 1)

        corners = tf.concat([top_left, top_right, bottom_left, bottom_right], axis=1)  # (batch,4)
        corner_mean = tf.reduce_mean(corners, axis=1, keepdims=True)  # (batch,1)

        corner_mean_expanded = tf.reshape(corner_mean, [batch_size, 1, 1, 1])
        return corner_mean_expanded

    def fill_black_with_mean(self, images, corner_mean_expanded):
        # Identify black pixels (exact match or use a threshold if needed)
        mask = tf.equal(images, 0.0)
        mask = tf.cast(mask, tf.float32)

        # Replace black pixels with the pre-computed corner mean
        images_filled = images * (1 - mask) + corner_mean_expanded * mask
        return images_filled

    def random_scale(self, images):
        # Random zoom by scaling in the range [0.9, 1.1]
        batch_size = tf.shape(images)[0]
        scale_factors = tf.random.uniform([batch_size], 0.9, 1.1)

        def scale_image(img, scale):
            shape = tf.shape(img)
            h, w = shape[0], shape[1]
            new_h = tf.cast(tf.floor(tf.cast(h, tf.float32)*scale), tf.int32)
            new_w = tf.cast(tf.floor(tf.cast(w, tf.float32)*scale), tf.int32)

            img_resized = tf.image.resize(img, (new_h, new_w))
            img_cropped = tf.image.resize_with_crop_or_pad(img_resized, h, w)
            return img_cropped

        images_list = tf.map_fn(
            lambda x: scale_image(x[0], x[1]),
            (images, scale_factors),
            fn_output_signature=tf.float32
        )
        return images_list

    def random_brightness(self, images):
        # Adjust brightness by a random factor in range [-0.2, 0.2]
        images = tf.image.random_brightness(images, max_delta=0.2)
        return images

    def random_focus(self, images):
        # Randomly choose to blur or not using tfa's gaussian_filter2d
        batch_size = tf.shape(images)[0]
        blur_prob = tf.random.uniform([batch_size], 0, 1) > 0.5

        def maybe_blur(img, apply_blur):
            return tf.cond(
                apply_blur,
                lambda: tfa.image.gaussian_filter2d(img, filter_shape=(3,3), sigma=1.0),
                lambda: img
            )

        images = tf.map_fn(
            lambda x: maybe_blur(x[0], x[1]),
            (images, blur_prob),
            fn_output_signature=tf.float32
        )
        return images
