
# utils/io.py
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


def load_images_from_mrcs(mrcs_file_path, txt_ids_path=None):
    """Load images from the specified .mrcs file and preprocess them, returning image names and preprocessed images."""
    image_names = []

    if os.path.exists(txt_ids_path):
        image_names.extend(read_list_from_txt(txt_ids_path))

    with mrcfile.open(mrcs_file_path, permissive=True) as mrc:
        # Read the data (it's a stack of 2D images)
        img_stack = mrc.data

    return img_stack, image_names