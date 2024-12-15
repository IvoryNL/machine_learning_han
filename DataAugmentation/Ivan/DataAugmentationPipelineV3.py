import tensorflow as tf

# Define augmentation pipeline
def create_augmentation_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(67, 50),  # Resize first
        tf.keras.layers.RandomRotation(0.1),  # Apply random rotation
        tf.keras.layers.RandomZoom(0.1, 0.1),  # Apply random zoom
        tf.keras.layers.RandomTranslation(0.1, 0.1),  # Apply random translation
        tf.keras.layers.Rescaling(1./255),  # Min-max scaling to [0, 1]
        tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),  # Convert to grayscale
    ])

# Load dataset dynamically
def load_dataset(source_folder, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        source_folder,
        image_size=(67, 50),  # Ensures resizing if needed
        batch_size=batch_size,
        color_mode='rgb',  # Load as RGB for flexibility
    )
    augmentation_pipeline = create_augmentation_pipeline()
    return dataset.map(lambda x, y: (augmentation_pipeline(x), y))

def process_pipeline():
    # source_folder = r"D:\Data\0. Machine Learning\0. Mini_project_finger_counting\0. Data\1. New data\Dataset Ivan"
    source_folder = input("Enter the source folder path: ")
    batch_size = int(input("Enter the batch size: "))

    dataset = load_dataset(source_folder, batch_size)
    return dataset