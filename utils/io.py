
# utils/io.py
from image_processing import preprocess_image
import mrcfile
import os


def create_directory(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def read_list_from_txt(filename):
    """
    Reads the contents of a .txt file and returns them as a list of strings.

    Parameters:
    filename (str): The name of the file to read.

    Returns:
    list: A list containing each line from the file as a separate item.
    """
    with open(filename, 'r') as file:
        data_list = file.read().splitlines()

    return data_list

def load_and_preprocess_images_from_mrcs(mrcs_file_path, txt_ids_path=None, target_size=(128, 128)):
    """Load images from the specified .mrcs file and preprocess them, returning image names and preprocessed images."""
    images = []
    image_names = []

    if os.path.exists(txt_ids_path):
        image_names.extend(read_list_from_txt(txt_ids_path))

    with mrcfile.open(mrcs_file_path, permissive=True) as mrc:
        # Read the data (it's a stack of 2D images)
        img_stack = mrc.data

        # Iterate over each 2D image in the stack
        for idx, img in enumerate(img_stack):
            # Preprocess the image (resize, normalize, etc.)
            preprocessed_image = preprocess_image(img, target_size=target_size)
            # Append the preprocessed image to the list
            images.append(preprocessed_image)

            if not os.path.exists(txt_ids_path):
                # Create a name for each image based on index or a naming convention
                filename = f"image_{idx + 1}"  # You can modify this as needed
                image_names.append(filename)

    return images, image_names