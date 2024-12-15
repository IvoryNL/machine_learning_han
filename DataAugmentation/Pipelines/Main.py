import os
import cv2
import tensorflow as tf
from PreprocessingPipelineV2 import PreprocessingPipeline

def save_augmented_images(dataset, output_folder):
    """
    Save augmented images to the file system for visualization using OpenCV.

    Args:
        dataset (tf.data.Dataset): The preprocessed dataset.
        output_folder (str): Path to save augmented images.
    """
    for augmented_images, labels in dataset:  # Each image now has multiple augmentations
        for i in range(len(labels)):
            label = labels[i].numpy()
            label_folder = os.path.join(output_folder, str(label), "Augmented")
            os.makedirs(label_folder, exist_ok=True)

            # Save each augmented image
            for j, augmented_image in enumerate(augmented_images[i]):
                # Ensure image format is compatible with OpenCV
                augmented_image = tf.squeeze(augmented_image).numpy()  # Remove extra dimensions

                # Scale the image to [0, 255] for OpenCV
                augmented_image = (augmented_image * 255).astype("uint8")

                # Save the image using OpenCV
                augmented_image_path = os.path.join(label_folder, f"augmented_{i + 1}_{j + 1}.png")
                cv2.imwrite(augmented_image_path, augmented_image)



def main():
    source_path = r"D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset Ivan V3"  # Path to the source images folder
    output_folder = source_path  # Save augmented images in the same base folder
    augmented_per_image = 10  # Number of augmented images per original image

    # Initialize preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline()

    # Process the dataset
    dataset = preprocessing_pipeline.process_dataset(source_path)

    # Save augmented images for testing
    save_augmented_images(dataset, output_folder)

if __name__ == "__main__":
    main()

















# import os
# import cv2
# import numpy as np
# from PreprocessingPipelineV1 import PreprocessingPipeline
#
# source_folder = r"D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset Ivan V3"
# batch_size = 64
# desired_augmented_images_per_label = 100  # Adjust as needed
#
# # Initialize pipeline
# pipeline = PreprocessingPipeline()
#
# augmented_dataset = pipeline.process(source_folder, desired_augmented_images_per_label)
#
# # # Process each label subfolder
# # for label_folder in os.listdir(source_folder):
# #     label_path = os.path.join(source_folder, label_folder)
# #     if not os.path.isdir(label_path):
# #         continue
# #
# #     # Augmented folder within each label directory
# #     augmented_folder = os.path.join(label_path, "Augmented")
# #     os.makedirs(augmented_folder, exist_ok=True)
# #
# #     # Get augmented dataset
# #     augmented_dataset = pipeline.process(label_path, desired_augmented_images_per_label)
# #
# #     # Print augmented_dataset for debugging
# #     print("Augmented dataset for label folder '{}':".format(label_folder))
# #     print(augmented_dataset)
# #     print("Type of augmented_dataset:", type(augmented_dataset))
# #     if isinstance(augmented_dataset, list):
# #         print("Length of augmented_dataset:", len(augmented_dataset))
# #         if len(augmented_dataset) > 0 and isinstance(augmented_dataset[0], list):
# #             print("Length of each inner list:", [len(sublist) for sublist in augmented_dataset])
# #
# #     # Save augmented images
# #     for i, image_list in enumerate(augmented_dataset):
# #         for j, image_data in enumerate(image_list):
# #             # Construct filename with prefix and iterator
# #             image_name = f"Ivan_{label_folder}_{i}_{j}.jpg"
# #             image_path = os.path.join(augmented_folder, image_name)
# #
# #             # Convert the float list to a numpy array and scale to 0-255
# #             # Assuming `image_data` is a 2D or 3D float array with values in [0,1]
# #             image_array = (np.array(image_data) * 255).astype(np.uint8)
# #
# #             # Save the image
# #             cv2.imwrite(image_path, image_array)