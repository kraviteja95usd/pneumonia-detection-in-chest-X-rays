from pathlib import Path

from matplotlib import pyplot as plt

from pneumonia_detector_constants import pneumonia_detector_constants
from PIL import Image
import matplotlib.image as mpimg
import cv2

import os
import pandas as pd

import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


class ImageDataPreprocessing:
    def __init__(self):
        """
            Initialize ImageDataPreprocessing class
        """
        super().__init__()
        self.chest_xray_folder_name = pneumonia_detector_constants["chest_xray_folder_name"]
        self.dataset_dir_name = pneumonia_detector_constants["dataset_dir_name"]
        self.image_information_dir_name = pneumonia_detector_constants["image_information_dir_name"]
        self.image_info_xls_file_name = pneumonia_detector_constants["image_info_xls_file_name"]
        self.train_test_image_dirs = pneumonia_detector_constants["train_test_image_dirs"]
        self.normal_pneumonia_image_dirs = pneumonia_detector_constants["normal_pneumonia_image_dirs"]

    def get_base_path_of_dataset(self):
        """
        Fetches the path to the dataset

        Args:
            None

        Returns:
            Folder name of the Chest X-ray image dataset
        """
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), str(self.dataset_dir_name))
        dataset_folder = os.path.join(dataset_path, str(self.chest_xray_folder_name))
        return dataset_folder

    def fetch_images_info_excel_file_path(self):
        """
        Fetches the absolute path & name of the Excel file where the images info is saved.

		Args:
		    None

		Returns:
			Absolute path of the Excel file
        """
        excel_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), str(self.image_information_dir_name))
        excel_file = os.path.join(excel_file_path, str(self.image_info_xls_file_name))
        return excel_file

    def count_images(self, dataset_path):
        """
        Capture the count of images in each folder/subfolder and Prints them

        Args:
            dataset_path(str): Absolute path of the image dataset directory from the get_base_path_of_dataset function

        Returns:
            None
        """

        if "_nrm" in dataset_path:
            train_normal = os.path.join(dataset_path, str(self.train_test_image_dirs[0]) + "_nrm", str(self.normal_pneumonia_image_dirs[0]) + "_nrm")
            train_pneumonia = os.path.join(dataset_path, str(self.train_test_image_dirs[0]) + "_nrm", str(self.normal_pneumonia_image_dirs[1]) + "_nrm")
            test_normal = os.path.join(dataset_path, str(self.train_test_image_dirs[1]) + "_nrm", str(self.normal_pneumonia_image_dirs[0]) + "_nrm")
            test_pneumonia = os.path.join(dataset_path, str(self.train_test_image_dirs[1]) + "_nrm", str(self.normal_pneumonia_image_dirs[1]) + "_nrm")
        else:
            train_normal = os.path.join(dataset_path, str(self.train_test_image_dirs[0]), str(self.normal_pneumonia_image_dirs[0]))
            train_pneumonia = os.path.join(dataset_path, str(self.train_test_image_dirs[0]), str(self.normal_pneumonia_image_dirs[1]))
            test_normal = os.path.join(dataset_path, str(self.train_test_image_dirs[1]), str(self.normal_pneumonia_image_dirs[0]))
            test_pneumonia = os.path.join(dataset_path, str(self.train_test_image_dirs[1]), str(self.normal_pneumonia_image_dirs[1]))

        print(f"Train Normal Path: {str(train_normal)}")
        print(f"\nTrain Pneumonia Path: {str(train_pneumonia)}")
        print(f"\nTest Normal Path: {str(test_normal)}")
        print(f"\nTest Pneumonia Path: {str(test_pneumonia)}")
        print("\n\n")

        train_normal_count = len(
            [name for name in os.listdir(train_normal) if
             os.path.isfile(os.path.join(train_normal, name)) and not
             name.startswith('.')]
        )
        train_pneumonia_count = len(
            [name for name in os.listdir(train_pneumonia) if os.path.isfile(os.path.join(train_pneumonia, name)) and
             not name.startswith('.')]
        )
        test_normal_count = len(
            [name for name in os.listdir(test_normal) if os.path.isfile(os.path.join(test_normal, name)) and
             not name.startswith('.')]
        )
        test_pneumonia_count = len(
            [name for name in os.listdir(test_pneumonia) if os.path.isfile(os.path.join(test_pneumonia, name)) and
             not name.startswith('.')]
        )

        return train_normal_count, train_pneumonia_count, test_normal_count, test_pneumonia_count

    @staticmethod
    def list_and_print_folders_tree(dataset_path):
        """
        Figure out the folders / sub-folders from a given path.
        Prints Tree structure of the dataset folder if exists
        Else it prints the provided dataset path is not valid

        Args:
            dataset_path (str): The absolute path of the dataset

        Returns:
            None
        """

        def list_folders_tree(image_dataset_path, level=0):
            """
            Recursively prints the directory tree structure.

            Args:
                image_dataset_path (str): The current folder to list out folders inside it.
                level (int): The current level in the directory tree.

            Returns:
                Tree structure of the given dataset folder
            """

            prefix = "|   " * (level - 1) + "|-- " if level > 0 else ""
            print(f"{prefix}{os.path.basename(image_dataset_path)}")

            for item in os.listdir(image_dataset_path):
                item_path = os.path.join(image_dataset_path, item)
                if os.path.isdir(item_path):
                    list_folders_tree(item_path, level + 1)

        if os.path.isdir(dataset_path):
            list_folders_tree(dataset_path)
        else:
            print(f"The path {dataset_path} is not a valid directory.")

    @staticmethod
    def get_image_info(image_path):
        """
        Read image and return its height, width, min value span and max value span

        Args:
            image_path: Path to the image

        Returns:
             Tuple of height and width
             Tuple of min value span and max value span
        """

        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, None
        # Get the dimensions of the image
        height, width = image.shape
        # Get the grayscale span
        min_val = image.min()
        max_val = image.max()
        return (height, width), (min_val, max_val)

    def read_images_and_capture_info_to_excel(self, image_dataset_path, level=0):
        """
        Read subfolders and save image information to an Excel file.

        Args:
            image_dataset_path (str): Path to the root directory of the image dataset.
            level (int, optional): Depth level for nested folders. Defaults to 0.

        Returns:
            None
        """

        dataset_folder = os.path.basename(os.path.normpath(image_dataset_path))

        # Check if the given path is a directory
        if not os.path.isdir(image_dataset_path):
            print(f"The path {image_dataset_path} is not a valid directory.")
            return

        data_dict = {}

        def read_images_from_subfolders(current_path, depth, folder_name):
            """
            Recursively read subfolders and collect image information (height, weight).

            Args:
                current_path (str): Path to the current directory being read.
                depth (int): Current depth level in the directory tree.
                folder_name (str): Name of the current folder.
            """

            # List all entries in the current directory
            with os.scandir(current_path) as entries:
                folder_data = []
                for entry in entries:
                    if entry.is_dir():
                        # Update the folder name for nested structure
                        new_folder_name = f"{folder_name}_{entry.name}"

                        # Recurse into the subdirectory
                        read_images_from_subfolders(entry.path, depth + 1, new_folder_name)
                    elif entry.is_file():
                        # Get image information
                        image_path = entry.path
                        size, span = self.get_image_info(image_path)

                        if size and span:
                            folder_data.append({
                                "File Name": entry.name,
                                "Height": size[0],
                                "Width": size[1],
                                "Grayscale Span1": span[0],
                                "Grayscale Span2": span[1],
                                "Image Path": current_path
                            })
                if folder_data:
                    data_dict[folder_name] = folder_data

        # Start reading images from subfolders using the given path
        read_images_from_subfolders(image_dataset_path, level, dataset_folder)

        # Save data to an Excel file with separate sheets
        excel_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), str(self.image_information_dir_name))
        excel_file_name = os.path.join(excel_file_path, str(self.image_info_xls_file_name))

        try:
            with pd.ExcelWriter(excel_file_name) as writer:
                for sheet_name, data in data_dict.items():
                    # Ensure the sheet name is within the 31-character limit. Keep the last 31 characters
                    valid_sheet_name = sheet_name[-31:]
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=valid_sheet_name, index=False)
            print(f"Image information is saved to {excel_file_name}")
        except PermissionError:
            print(f"Permission denied: Unable to write to {excel_file_name}")
            print("Ensure the file is not open and you have write permissions")

    def display_first_image(self, folder_path):
        # Check if the folder path exists and is a directory
        if not os.path.exists(folder_path):
            print(f"Error: The folder path '{folder_path}' does not exist.")
            return
        if not os.path.isdir(folder_path):
            print(f"Error: The path '{folder_path}' is not a directory.")
            return

        try:
            # Get a list of all image files in the folder
            image_files = [file for file in os.listdir(folder_path)
                           if file.endswith('.jpg') or
                           file.endswith('.png') or
                           file.endswith('.jpeg')]

            # Check if there are any image files in the folder
            if image_files:
                # Open the first image file
                image_path = os.path.join(folder_path, image_files[0])
                image = mpimg.imread(image_path)

                # Display the image
                plt.imshow(image)
                plt.axis('off')  # Turn off axis
                plt.show()
            else:
                print("No image files found in the folder.")
        except OSError as ex:
            print(f"Error: {ex}")

    def list_and_create_folders_tree(self, image_dataset_path, new_image_dataset_path, level=0):
        """
        Recursively lists and creates a mirrored folder tree with '_nrm' appended to folder names.
        This is to save the normalized images inside them

        Args:
            image_dataset_path (str): The path of the root folder to mirror.
            new_image_dataset_path (str): The path where the new folder structure will be created.
            level (int): The current level in the folder hierarchy (used for display purposes).

        Returns:
            None
        """

        # Display the folder hierarchy
        prefix = "|   " * (level - 1) + "|-- " if level > 0 else ""
        # print(f"{prefix}{os.path.basename(root_folder)}")

        # Create the new folder structure with '_nrm' appended to the names
        new_folder_name = os.path.basename(image_dataset_path) + "_nrm"
        new_folder_path = os.path.join(new_image_dataset_path, new_folder_name)
        Path(new_folder_path).mkdir(parents=True, exist_ok=True)

        for item in os.listdir(image_dataset_path):
            item_path = os.path.join(image_dataset_path, item)
            if os.path.isdir(item_path):
                self.list_and_create_folders_tree(item_path, new_folder_path, level + 1)

    @staticmethod
    def normalize_image(image):
        """
        Normalize the pixel values of a grayscale image to the range [0, 255].

        Array:
            image (numpy.ndarray): The input image as a 2D array of pixel values.

        Returns:
            numpy.ndarray: The normalized image with pixel values in the range [0, 255] as an unsigned 8-bit integer array.
        """
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:  # Avoid division by zero
            normalized = (image - min_val) / (max_val - min_val) * 255
            return normalized.astype('uint8')
        else:
            return image

    def normalize_images_in_folders(self, image_dataset_path):
        """
        Normalize images in the specified directory and save them to a new directory.

        Args:
            image_dataset_path (str): The path to the directory containing images (original image dataset).

        Returns:
            None
        """

        nrm_image_dataset_path = image_dataset_path + "_nrm"

        if not os.path.isdir(image_dataset_path):
            print(f"The path {image_dataset_path} is not a valid directory.")
            return

        def process_subfolders(current_path, current_output_path):
            """
            Recursively process subfolders to normalize images.

            Args:
                current_path (str): The current directory path being processed (original image dataset).
                current_output_path (str): The output directory path where normalized images will be saved.

            Returns:
                None
            """
            with os.scandir(current_path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        new_output_path = os.path.join(current_output_path, entry.name + "_nrm")
                        process_subfolders(entry.path, new_output_path)
                    elif entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        image_path = entry.path
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            normalized_image = self.normalize_image(image)
                            Path(current_output_path).mkdir(parents=True, exist_ok=True)
                            normalized_image_path = os.path.join(current_output_path, entry.name)
                            cv2.imwrite(normalized_image_path, normalized_image)
                        else:
                            print(f"Failed to read image: {image_path}")

        process_subfolders(image_dataset_path, nrm_image_dataset_path)

    @staticmethod
    def load_excel_and_fetch_max_dimensions_of_images(excel_file_path):
        """
        Loads Excel file.
        Loop through all sheets in an Excel file and determine the maximum values for 'Height' and 'Width' columns.

        Args:
            excel_file_path (str): Path to the Excel file.

        Returns:
            tuple: Maximum values for 'Height' and 'Width' columns.
        """

        # Load the Excel file into an object
        xlsx_object = pd.ExcelFile(excel_file_path)

        # Initialize variables to keep track of the largest sizes
        max_height = 0
        max_width = 0

        # Loop through all sheets in the Excel file
        for sheet_name in xlsx_object.sheet_names:
            df = pd.read_excel(xlsx_object, sheet_name=sheet_name)
            max_height = max(max_height, df['Height'].max())
            max_width = max(max_width, df['Width'].max())

        return max_height, max_width

    @staticmethod
    def resize_and_pad_image(image_path, output_path, height_size, width_size):
        """
        Resizes and pads an image to the specified height and width, saving it to the output path.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the resized and padded image.
            height_size (int): Desired height of the new image.
            width_size (int): Desired width of the new image.

        Returns:
            None
        """

        with Image.open(image_path) as img:
            # Get the original width and height of the image
            width, height = img.size

            # Define the new image dimensions
            new_width = width_size
            new_height = height_size

            # Create a new image with the specified dimensions and a black background
            new_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))

            # Calculate the position to paste the old image onto the new image
            left = (new_width - width) // 2
            top = (new_height - height) // 2

            # Paste the original image onto the new image at the calculated position
            new_img.paste(img, (left, top))

            # Save the new image with appropriate quality settings based on the file extension
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                new_img.save(output_path, format='JPEG', quality=95)  # Adjust quality as needed
            else:
                new_img.save(output_path)  # For PNG or other formats

    def process_directory(self, nrm_images_folder_path, height, width):
        """
        Processes all images in a directory, resizing and padding each image to the specified dimensions.

        Args:
            nrm_images_folder_path (str): Path to the root folder containing images.
            height (int): Desired height for all images.
            width (int): Desired width for all images.

        Returns:
            None
        """

        for subdir, _, files in os.walk(nrm_images_folder_path):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    output_path = os.path.join(subdir, f"{file}")
                    self.resize_and_pad_image(file_path, output_path, height, width)
                    print(f"Processed image: {file_path}")
