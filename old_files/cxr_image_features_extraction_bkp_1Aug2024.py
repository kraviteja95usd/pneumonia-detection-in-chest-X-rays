import pathlib
from os import getcwd
from sys import path

from tqdm import tqdm

path.append(getcwd() + "/bin")

from image_pre_processing import ImageDataPreprocessing
from joblib import Parallel, delayed
from pneumonia_detector_constants import pneumonia_detector_constants
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops

import cv2
import os
import numpy as np
import pandas as pd

import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


class CxrImageFeatureExtraction(ImageDataPreprocessing):
	def __init__(self):
		"""
            Initialize CxrImageFeatureExtraction class
        """
		super().__init__()
		self.excel_sheet_names = pneumonia_detector_constants["excel_sheet_names"]
		self.image_first_order_features_xls_file_name = pneumonia_detector_constants["image_first_order_features_xls_file_name"]
		self.image_second_order_features_glcm_xls_file_name = pneumonia_detector_constants["image_second_order_features_glcm_xls_file_name"]
		self.image_second_order_features_glrlm_xls_file_name = pneumonia_detector_constants["image_second_order_features_glrlm_xls_file_name"]
		self.image_second_order_features_gldm_xls_file_name = pneumonia_detector_constants["image_second_order_features_gldm_xls_file_name"]
		self.image_second_order_features_ngtdm_xls_file_name = pneumonia_detector_constants["image_second_order_features_ngtdm_xls_file_name"]

	def fetch_images_features_excel_file_path(self):
		"""
		Fetches the absolute path & name of the Excel file where the images info is saved.

		Args:
			None

		Returns:
			Absolute path of the Excel file
		"""

		excel_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), str(self.image_information_dir_name))
		excel_file = os.path.join(excel_file_path, str(self.image_first_order_features_xls_file_name))
		return excel_file

	@staticmethod
	def fetch_last_two_folder_names_of_path(folder_path):
		path_obj = pathlib.Path(folder_path)
		last_two_folder_names = f"{path_obj.parent.name}/{path_obj.name}"
		return last_two_folder_names

	@staticmethod
	def extract_first_order_features(image):
		"""
		Calculate first-order features of an image.

		First-order features are statistics calculated from the pixel values of an image.
		They provide information about the distribution of pixel values, such as mean, median, standard deviation,
		variance, skewness, kurtosis, range, entropy, energy and uniformity

		Args:
			image (numpy array): Input image.

		Returns:
			first_order_features(dict): Dictionary containing the calculated first-order features.
		"""

		# Initialize an empty dictionary to store the features
		first_order_features = {}

		# Calculate the mean pixel intensity of the image
		# Mean: The average value of all pixel values in the image
		first_order_features["Mean"] = np.mean(image)  # np.mean() calculates the mean of the array

		# Calculate the median pixel intensity of the image
		# Median: The middle value of the sorted pixel values
		first_order_features["Median"] = np.median(image)  # np.median() calculates the median of the array

		# Calculate the standard deviation of pixel intensities
		# Standard Deviation: A measure of the spread of pixel values from the mean
		first_order_features["Standard Deviation"] = np.std(
			image)  # np.std() calculates the standard deviation of the array

		# Calculate the variance of pixel intensities
		# Variance: The average of the squared differences from the mean
		first_order_features["Variance"] = np.var(image)  # np.var() calculates the variance of the array

		# Calculate the skewness (asymmetry) of the pixel intensity distribution
		# Skewness: A measure of the asymmetry of the distribution
		first_order_features["Skewness"] = skew(image, axis=None)  # skew() calculates the skewness of the array

		# Calculate the kurtosis of the pixel intensity distribution
		# Kurtosis: A measure of the "tailedness" of the distribution
		first_order_features["Kurtosis"] = kurtosis(image, axis=None)  # kurtosis() calculates the kurtosis of the array

		# Calculate the range of pixel intensities (max - min)
		# Range: The difference between the maximum and minimum pixel values
		first_order_features["Range"] = np.ptp(image)  # np.ptp() calculates the range of the array

		# Calculate the entropy (measure of randomness) of the pixel intensities
		# Entropy: A measure of the unpredictability of the pixel values
		first_order_features["Entropy"] = entropy(image.ravel())  # entropy() calculates the entropy of the array

		# Calculate the energy (sum of squared pixel values)
		# Energy: A measure of the sum of squared pixel values
		# High energy: An image with many high-intensity pixels
		first_order_features["Energy"] = np.sum(image ** 2)  # np.sum() calculates the sum of the squared array

		# Calculate the uniformity (sum of squared normalized pixel values)
		# Uniformity: A measure of the sum of squared normalized pixel values
		# High uniformity: An image with many similar pixel values

		# np.sum() calculates the sum of the squared normalized array
		first_order_features["Uniformity"] = np.sum((image / 255.0) ** 2)

		# Return the calculated features as a dictionary
		return first_order_features

	# Extract GLCM features from an image
	@staticmethod
	def extract_glcm_features(image):
		"""
		Extracts GLCM (Gray-Level Co-occurrence Matrix) features from the given image.

		Args:
			image (numpy.ndarray): The input image (numpy array representing the input image as an argument).

		Returns:
			glcm_features(dict): A dictionary containing the extracted GLCM features.
		"""

		# Create a Gray-Level Co-occurrence Matrix (GLCM) from the input image
		# with distances of 1 pixel, at angles of 0, 45, 90, and 135 degrees
		# and symmetric and normalized properties
		glcm = graycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)

		# Extract the contrast feature from the GLCM
		contrast = graycoprops(glcm, 'contrast')

		# Extract the correlation feature from the GLCM
		correlation = graycoprops(glcm, 'correlation')

		# Extract the energy feature from the GLCM
		energy = graycoprops(glcm, 'energy')

		# Extract the homogeneity feature from the GLCM
		homogeneity = graycoprops(glcm, 'homogeneity')

		# Calculate the average contrast over all directions
		contrast = np.mean(contrast)

		# Calculate the average correlation over all directions
		correlation = np.mean(correlation)

		# Calculate the average energy over all directions
		energy = np.mean(energy)

		# Calculate the average homogeneity over all directions
		homogeneity = np.mean(homogeneity)

		# Create a dictionary to store the extracted GLCM features
		glcm_features = {
			'GLCM_Contrast': contrast,
			'GLCM_Correlation': correlation,
			'GLCM_Energy': energy,
			'GLCM_Homogeneity': homogeneity
		}

		# Return the dictionary of extracted GLCM features
		return glcm_features

	# Extract GLRLM Features from an image
	@staticmethod
	def extract_glrlm_features(image):
		"""
		Extracts GLRLM (Gray Level Run Length Matrix) features from the given image.

		Args:
			image (numpy.ndarray): The input image (numpy array representing the input image as an argument).

		Returns:
			glrlm_features(dict): A dictionary containing the extracted GLCM features.
		"""

		# Evaluate GLRLM
		# Get the maximum gray level in the image
		max_gray_level = np.max(image)

		# Initialize the GLRLM matrix with zeros
		# The size of the matrix is (max_gray_level + 1) x (image.shape[0] + image.shape[1] - 1)
		glrlm = np.zeros((max_gray_level + 1, image.shape[0] + image.shape[1] - 1))

		# Iterate over each pixel in the image
		for row_index in range(image.shape[0]):
			for column_index in range(image.shape[1]):
				# Get the gray level of the current pixel
				gray_level = image[row_index, column_index]

				# Initialize the run length to 1
				run_length = 1

				# Check if the next pixel has the same gray level
				while column_index + 1 < image.shape[1] and image[row_index, column_index + 1] == gray_level:
					# If it does, increment the column index and run length
					column_index += 1
					run_length += 1

				# Increment the corresponding element in the GLRLM matrix
				glrlm[gray_level, run_length - 1] += 1

		# Extract the required GLRLM features
		# Calculate the short run emphasis (SRE)
		# SRE is the sum of the elements in the first two columns of the GLRLM matrix
		# divided by the sum of all elements in the GLRLM matrix
		short_run_emphasis = np.sum(glrlm[:, 0:2]) / np.sum(glrlm)

		# Calculate the long run emphasis (LRE)
		# LRE is the sum of the elements in the last two columns of the GLRLM matrix
		# divided by the sum of all elements in the GLRLM matrix
		long_run_emphasis = np.sum(glrlm[:, -2:]) / np.sum(glrlm)

		# Calculate the gray level non-uniformity (GLN)
		# GLN is the sum of the squares of the row sums of the GLRLM matrix
		# divided by the sum of all elements in the GLRLM matrix
		gray_level_nonuniformity = np.sum(np.sum(glrlm, axis=1) ** 2) / np.sum(glrlm)

		# Calculate the run length non-uniformity (RLN)
		# RLN is the sum of the squares of the column sums of the GLRLM matrix
		# divided by the sum of all elements in the GLRLM matrix
		run_length_nonuniformity = np.sum(np.sum(glrlm, axis=0) ** 2) / np.sum(glrlm)

		# Create a dictionary to store the extracted GLRLM features
		glrlm_features = {
			'GLRLM_ShortRunEmphasis': short_run_emphasis,
			'GLRLM_LongRunEmphasis': long_run_emphasis,
			'GLRLM_GrayLevelNonuniformity': gray_level_nonuniformity,
			'GLRLM_RunLengthNonuniformity': run_length_nonuniformity
		}

		# Return the dictionary of extracted GLRLM features
		return glrlm_features

	# Extract GLDM Features from an image
	@staticmethod
	def extract_gldm_features(image, distance=1):
		"""
		Extracts GLDM (Gray Level Dependence Matrix) features from the given image.

		Args:
			image (numpy.ndarray): The input image (numpy array representing the input image as an argument).
			distance (int, optional): The distance within which neighboring pixels are considered for calculating the GLDM features. Defaults to 1.

		Returns:
			gldm_features(dict): A dictionary containing the extracted GLCM features.
		"""

		# Evaluate GLDM
		# Get the maximum gray level in the image
		max_gray_level = np.max(image)

		# Initialize the GLDM matrix with zeros
		# The size of the matrix is (max_gray_level + 1) x (max_gray_level + 1)
		gldm = np.zeros((max_gray_level + 1, max_gray_level + 1))

		# Iterate over each pixel in the image
		for row_index in range(image.shape[0]):
			for column_index in range(image.shape[1]):
				# Get the gray level of the current pixel
				gray_level = image[row_index, column_index]

				# Iterate over the neighboring pixels within the specified distance
				for row_offset in range(-distance, distance + 1):
					for column_offset in range(-distance, distance + 1):
						# Skip the current pixel itself
						if row_offset == 0 and column_offset == 0:
							continue

						# Calculate the row and column indices of the neighboring pixel
						neighbor_row_index = row_index + row_offset
						neighbor_column_index = column_index + column_offset

						# Check if the neighboring pixel is within the image boundaries
						if (neighbor_row_index < 0 or
								neighbor_row_index >= image.shape[0] or
								neighbor_column_index < 0 or
								neighbor_column_index >= image.shape[1]):
							continue

						# Get the gray level of the neighboring pixel
						neighbor_gray_level = image[neighbor_row_index, neighbor_column_index]

						# Increment the corresponding element in the GLDM matrix
						gldm[gray_level, neighbor_gray_level] += 1

		# Extract the required GLDM features
		# Calculate the small dependence emphasis (SDE)
		# SDE is the sum of the elements in the first two rows of the GLDM matrix
		# divided by the sum of all elements in the GLDM matrix
		small_dependence_emphasis = np.sum(gldm[0:2, :]) / np.sum(gldm)

		# Calculate the large dependence emphasis (LDE)
		# LDE is the sum of the elements in the last two rows of the GLDM matrix
		# divided by the sum of all elements in the GLDM matrix
		large_dependence_emphasis = np.sum(gldm[-2:, :]) / np.sum(gldm)

		# Calculate the gray level non-uniformity (GLN)
		# GLN is the sum of the squares of the row sums of the GLDM matrix
		# divided by the sum of all elements in the GLDM matrix
		gray_level_non_uniformity = np.sum(np.sum(gldm, axis=1) ** 2) / np.sum(gldm)

		# Calculate the dependence count non-uniformity (DCN)
		# DCN is the sum of the squares of the column sums of the GLDM matrix
		# divided by the sum of all elements in the GLDM matrix
		dependence_count_non_uniformity = np.sum(np.sum(gldm, axis=0) ** 2) / np.sum(gldm)

		# Create a dictionary to store the extracted GLDM features
		gldm_features = {
			'GLDM_SmallDependenceEmphasis': small_dependence_emphasis,
			'GLDM_LargeDependenceEmphasis': large_dependence_emphasis,
			'GLDM_GrayLevelNonuniformity': gray_level_non_uniformity,
			'GLDM_DependenceCountNonuniformity': dependence_count_non_uniformity
		}

		# Return the dictionary of extracted GLDM features
		return gldm_features

	# Extract NGTDM Features from an image
	@staticmethod
	def extract_ngtdm_features(image):
		"""
		Extracts NGTDM (Neighborhood Gray Tone Difference Matrix) features from the given image.

		Args:
			image (numpy.ndarray): The input image (numpy array representing the input image as an argument).

		Returns:
			ngtdm_features(dict): A dictionary containing the extracted GLCM features.
		"""

		# Evaluate NGTDM
		# Get the maximum gray level in the image
		max_gray_level = np.max(image)

		# Initialize the NGTDM matrix with zeros
		# The size of the matrix is (max_gray_level + 1) x 1
		ngtdm = np.zeros((max_gray_level + 1, 1))

		# Iterate over each pixel in the image
		for row_index in range(image.shape[0]):
			for column_index in range(image.shape[1]):
				# Get the gray level of the current pixel
				gray_level = image[row_index, column_index]

				# Initialize an empty list to store the neighboring pixels
				neighborhood = []

				# Iterate over the neighboring pixels
				for row_offset in range(-1, 2):
					for column_offset in range(-1, 2):
						# Skip the current pixel itself
						if row_offset == 0 and column_offset == 0:
							continue

						# Calculate the row and column indices of the neighboring pixel
						neighbor_row_index = row_index + row_offset
						neighbor_column_index = column_index + column_offset

						# Check if the neighboring pixel is within the image boundaries
						if (neighbor_row_index < 0 or
								neighbor_row_index >= image.shape[0] or
								neighbor_column_index < 0 or
								neighbor_column_index >= image.shape[1]):
							continue

						# Append the neighboring pixel to the list
						neighborhood.append(image[neighbor_row_index, neighbor_column_index])

				# Convert the list of neighboring pixels to a numpy array
				neighborhood = np.array(neighborhood)

				# Calculate the sum of the absolute differences between the neighboring pixels and the current pixel
				ngtdm[gray_level, 0] += np.sum(np.abs(neighborhood - gray_level))

		# Extract the required NGTDM features
		# Calculate the coarseness feature
		# Coarseness is the sum of the NGTDM matrix divided by the total number of pixels in the image
		coarseness = np.sum(ngtdm) / (image.shape[0] * image.shape[1])

		# Calculate the contrast feature
		# Contrast is the sum of the squared NGTDM matrix divided by the total number of pixels in the image
		contrast = np.sum(ngtdm * ngtdm) / (image.shape[0] * image.shape[1])

		# Calculate the busyness feature
		# Busyness is the sum of the NGTDM matrix divided by the product of the total number of pixels in the image and
		# the sum of the absolute differences between the image pixels and the mean image pixel value
		busyness = np.sum(ngtdm) / (image.shape[0] * image.shape[1] * np.sum(np.abs(image - np.mean(image))))

		# Calculate the complexity feature
		# Complexity is the sum of the product of the NGTDM matrix and the logarithm of the NGTDM matrix plus 1,
		# divided by the total number of pixels in the image
		complexity = np.sum(ngtdm * np.log2(ngtdm + 1)) / (image.shape[0] * image.shape[1])

		# Create a dictionary to store the extracted NGTDM features
		ngtdm_features = {
			'NGTDM_Coarseness': coarseness,
			'NGTDM_Contrast': contrast,
			'NGTDM_Busyness': busyness,
			'NGTDM_Complexity': complexity
		}

		# Return the dictionary of extracted NGTDM features
		return ngtdm_features

	def extract_first_order_features_of_image(
			self, image_path: str, file: str) -> dict:
		"""
		Extracts all features of an image.

		Args:
			image_path (str): The path to the image file.
			file (str): The filename of the image.

		Returns:
			first_order_features(dict): A dictionary containing all the extracted first order features of the image.

		Raises:
			FileNotFoundError: If the image file is not found.
		"""

		# Read the image in grayscale mode
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		# Check if the image is loaded correctly
		if image is None:
			raise FileNotFoundError(f"Image file not found: {image_path}")

		# Extract First Order Features
		# Calculate the first order features of the image
		first_order_features = self.extract_first_order_features(image)

		first_order_features = {
			"Image Filename": file,
			**first_order_features
		}

		# Return the dictionary of extracted features
		return first_order_features

	def extract_second_order_glcm_features(
			self, image_path: str, file: str) -> dict:
		"""
		Extracts all features of an image.

		Args:
			image_path (str): The path to the image file.
			file (str): The filename of the image.

		Returns:
			extracted_glcm_features(dict): A dictionary containing all the extracted second order GLCM features of the image.

		Raises:
			FileNotFoundError: If the image file is not found.
		"""

		# Read the image in grayscale mode
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		# Check if the image is loaded correctly
		if image is None:
			raise FileNotFoundError(f"Image file not found: {image_path}")

		# Extract First Order Features
		# Calculate the first order features of the image
		extracted_glcm_features = self.extract_glcm_features(image)

		extracted_glcm_features = {
			"Image Filename": file,
			**extracted_glcm_features
		}

		# Return the dictionary of extracted features
		return extracted_glcm_features

	def extract_second_order_glrlm_features(
			self, image_path: str, file: str) -> dict:
		"""
		Extracts all features of an image.

		Args:
			image_path (str): The path to the image file.
			file (str): The filename of the image.

		Returns:
			extracted_glrlm_features(dict): A dictionary containing all the extracted second order GLRLM features of the image.

		Raises:
			FileNotFoundError: If the image file is not found.
		"""

		# Read the image in grayscale mode
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		# Check if the image is loaded correctly
		if image is None:
			raise FileNotFoundError(f"Image file not found: {image_path}")

		# Extract First Order Features
		# Calculate the first order features of the image
		extracted_glrlm_features = self.extract_glrlm_features(image)

		extracted_glrlm_features = {
			"Image Filename": file,
			**extracted_glrlm_features
		}

		# Return the dictionary of extracted features
		return extracted_glrlm_features

	def extract_second_order_gldm_features(
			self, image_path: str, file: str) -> dict:
		"""
		Extracts all features of an image.

		Args:
			image_path (str): The path to the image file.
			file (str): The filename of the image.

		Returns:
			extracted_gldm_features(dict): A dictionary containing all the extracted second order GLDM features of the image.

		Raises:
			FileNotFoundError: If the image file is not found.
		"""

		# Read the image in grayscale mode
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		# Check if the image is loaded correctly
		if image is None:
			raise FileNotFoundError(f"Image file not found: {image_path}")

		# Extract First Order Features
		# Calculate the first order features of the image
		extracted_gldm_features = self.extract_gldm_features(image)

		extracted_gldm_features = {
			"Image Filename": file,
			**extracted_gldm_features
		}

		# Return the dictionary of extracted features
		return extracted_gldm_features

	def extract_second_order_ngtdm_features(
			self, image_path: str, file: str) -> dict:
		"""
		Extracts all features of an image.

		Args:
			image_path (str): The path to the image file.
			file (str): The filename of the image.

		Returns:
			extracted_ngtdm_features(dict): A dictionary containing all the extracted second order GLDM features of the image.

		Raises:
			FileNotFoundError: If the image file is not found.
		"""

		# Read the image in grayscale mode
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

		# Check if the image is loaded correctly
		if image is None:
			raise FileNotFoundError(f"Image file not found: {image_path}")

		# Extract First Order Features
		# Calculate the first order features of the image
		extracted_ngtdm_features = self.extract_ngtdm_features(image)

		extracted_ngtdm_features = {
			"Image Filename": file,
			**extracted_ngtdm_features
		}

		# Return the dictionary of extracted features
		return extracted_ngtdm_features

	# def update_first_order_features_to_excel_file(
	# 		self, excel_file_name, folders):
	# 	excel_sheet_names = self.excel_sheet_names
	#
	# 	# Create an Excel writer object
	# 	with pd.ExcelWriter(excel_file_name) as writer:
	# 		for index, folder in enumerate(folders):
	# 			print(f"Extracting first-order features from: {folder}")
	# 			files = sorted(os.listdir(folder))
	# 			last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
	#
	# 			with tqdm(total=len(files), desc=f"Folder: {last_two_folder_names}") as pbar:
	# 				features = Parallel(n_jobs=-1)(
	# 					delayed(self.extract_first_order_features_of_image)(os.path.join(folder, file), file)
	# 					for file in files
	# 				)
	# 				pbar.update(len(files))
	#
	# 			first_order_features_df = pd.DataFrame(features)
	# 			first_order_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)
	#
	# 	print(f"\n\nAll first-order features are extracted to the Excel file: {excel_file_name}")
	# 	print("Please check the Excel file for further analysis and interpretation")

	# def update_first_order_features_to_excel_file(self, excel_file_name, folders):
	# 	excel_sheet_names = self.excel_sheet_names
	#
	# 	# Create an Excel writer object
	# 	with pd.ExcelWriter(excel_file_name) as writer:
	# 		for index, folder in enumerate(folders):
	# 			print(f"Extracting first-order features from: {folder}")
	# 			files = sorted(os.listdir(folder))
	# 			last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
	# 			with tqdm(total=len(files), desc=f"Folder: {last_two_folder_names}") as pbar:
	# 				features = Parallel(n_jobs=-1)(
	# 					delayed(self.extract_first_order_features_of_image)(os.path.join(folder, file), file)
	# 					for file in files
	# 				)
	# 				pbar.update(len(files))
	# 			first_order_features_df = pd.DataFrame(features)
	# 			first_order_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)
	#
	# 	print(f"\n\nAll first-order features are extracted to the Excel file: {excel_file_name}")
	# 	print("Please check the Excel file for further analysis and interpretation")

	def update_first_order_features_to_excel_file(self, excel_file_name, folders):
		excel_sheet_names = self.excel_sheet_names

		# Create an Excel writer object
		with pd.ExcelWriter(excel_file_name) as writer:
			for index, folder in enumerate(folders):
				print(f"Extracting first-order features from: {folder}")
				files = sorted(os.listdir(folder))
				last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
				features = Parallel(n_jobs=-1)(
					delayed(self.extract_first_order_features_of_image)(os.path.join(folder, file), file)
					for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
				)
				first_order_features_df = pd.DataFrame(features)
				first_order_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

		print(f"\n\nAll first-order features are extracted to the Excel file: {excel_file_name}")
		print("Please check the Excel file for further analysis and interpretation")

	# def update_second_order_glcm_features_to_excel_file(self, excel_file_name, folders):
	# 	excel_sheet_names = self.excel_sheet_names
	#
	# 	# Create an Excel writer object
	# 	with pd.ExcelWriter(excel_file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
	# 		for index, folder in enumerate(folders):
	# 			print(f"Extracting second-order GLCM features from: {folder}")
	# 			files = sorted(os.listdir(folder))
	# 			last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
	# 			with tqdm(total=len(files), desc=f"Folder: {last_two_folder_names}") as pbar:
	# 				features = Parallel(n_jobs=-1)(
	# 					delayed(self.extract_second_order_glcm_features)(os.path.join(folder, file), file)
	# 					for file in files
	# 				)
	# 				pbar.update(len(files))
	#
	# 			# Read the existing first order features from the Excel file
	# 			first_order_features_df = pd.read_excel(excel_file_name, sheet_name=excel_sheet_names[index])
	#
	# 			# Create a new DataFrame with the second order GLCM features
	# 			glcm_features_df = pd.DataFrame(features)
	#
	# 			# Merge the two DataFrames based on the 'Image Filename' column
	# 			combined_features_df = pd.merge(first_order_features_df, glcm_features_df, on='Image Filename')
	#
	# 			# Write the combined DataFrame to the Excel file
	# 			combined_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)
	#
	# 	print(f"\n\nAll second-order GLCM features are extracted to the Excel file: {excel_file_name}")
	# 	print("Please check the Excel file for further analysis and interpretation")

	def update_second_order_glcm_features_to_excel_file(self, excel_file_name, folders):
		excel_sheet_names = self.excel_sheet_names

		# Create an Excel writer object
		with pd.ExcelWriter(excel_file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
			for index, folder in enumerate(folders):
				print(f"Extracting second-order GLCM features from: {folder}")
				files = sorted(os.listdir(folder))
				last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
				features = Parallel(n_jobs=-1)(
					delayed(self.extract_second_order_glcm_features)(os.path.join(folder, file), file)
					for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
				)

				# Read the existing first order features from the Excel file
				first_order_features_df = pd.read_excel(excel_file_name, sheet_name=excel_sheet_names[index])

				# Create a new DataFrame with the second order GLCM features
				glcm_features_df = pd.DataFrame(features)

				# Merge the two DataFrames based on the 'Image Filename' column
				combined_features_df = pd.merge(first_order_features_df, glcm_features_df, on='Image Filename')

				# Write the combined DataFrame to the Excel file
				combined_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

		print(f"\n\nAll second-order GLCM features are extracted to the Excel file: {excel_file_name}")
		print("Please check the Excel file for further analysis and interpretation")

	# def update_second_order_glrlm_features_to_excel_file(
	# 		self, excel_file_name, folders):
	# 	excel_sheet_names = self.excel_sheet_names
	#
	# 	# Create an Excel writer object
	# 	with pd.ExcelWriter(excel_file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
	# 		for index, folder in enumerate(folders):
	# 			print(f"Extracting second-order GLRLM features from: {folder}")
	# 			files = sorted(os.listdir(folder))
	# 			last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
	# 			with tqdm(total=len(files), desc=f"Folder: {last_two_folder_names}") as pbar:
	# 				features = Parallel(n_jobs=-1)(
	# 					delayed(self.extract_second_order_glrlm_features)(os.path.join(folder, file), file)
	# 					for file in files
	# 				)
	# 				pbar.update(len(files))
	#
	# 			# Read the existing first order features from the Excel file
	# 			first_order_features_df = pd.read_excel(excel_file_name, sheet_name=excel_sheet_names[index])
	#
	# 			# Create a new DataFrame with the second order GLCM features
	# 			glrlm_features_df = pd.DataFrame(features)
	#
	# 			# Merge the two DataFrames based on the 'Image Filename' column
	# 			combined_features_df = pd.merge(first_order_features_df, glrlm_features_df, on='Image Filename')
	#
	# 			# Write the combined DataFrame to the Excel file
	# 			combined_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)
	#
	# 	print(f"\n\nAll second-order GLRLM features are extracted to the Excel file: {excel_file_name}")
	# 	print("Please check the Excel file for further analysis and interpretation")

	def update_second_order_glrlm_features_to_excel_file(
			self, excel_file_name, folders):
		excel_sheet_names = self.excel_sheet_names

		# Create an Excel writer object
		with pd.ExcelWriter(excel_file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
			for index, folder in enumerate(folders):
				print(f"Extracting second-order GLRLM features from: {folder}")
				files = sorted(os.listdir(folder))
				last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
				features = Parallel(n_jobs=-1)(
					delayed(self.extract_second_order_glrlm_features)(os.path.join(folder, file), file)
					for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
				)

				# Read the existing first order features from the Excel file
				first_order_features_df = pd.read_excel(excel_file_name, sheet_name=excel_sheet_names[index])

				# Create a new DataFrame with the second order GLRLM features
				glrlm_features_df = pd.DataFrame(features)

				# Merge the two DataFrames based on the 'Image Filename' column
				combined_features_df = pd.merge(first_order_features_df, glrlm_features_df, on='Image Filename')

				# Write the combined DataFrame to the Excel file
				combined_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

		print(f"\n\nAll second-order GLRLM features are extracted to the Excel file: {excel_file_name}")
		print("Please check the Excel file for further analysis and interpretation")

	def update_second_order_gldm_features_to_excel_file(
			self, excel_file_name, folders):
		excel_sheet_names = self.excel_sheet_names

		# Create an Excel writer object
		with pd.ExcelWriter(excel_file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
			for index, folder in enumerate(folders):
				print(f"Extracting second-order GLDM features from: {folder}")
				files = sorted(os.listdir(folder))
				last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
				features = Parallel(n_jobs=-1)(
					delayed(self.extract_second_order_gldm_features)(os.path.join(folder, file), file)
					for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
				)

				# with tqdm(total=len(files), desc=f"Folder: {last_two_folder_names}") as pbar:
				# 	features = Parallel(n_jobs=-1)(
				# 		delayed(self.extract_second_order_gldm_features)(os.path.join(folder, file), file)
				# 		for file in files
				# 	)
				# 	pbar.update()

				# Read the existing first order features from the Excel file
				first_order_features_df = pd.read_excel(excel_file_name, sheet_name=excel_sheet_names[index])

				# Create a new DataFrame with the second order GLCM features
				gldm_features_df = pd.DataFrame(features)

				# Merge the two DataFrames based on the 'Image Filename' column
				combined_features_df = pd.merge(first_order_features_df, gldm_features_df, on='Image Filename')

				# Write the combined DataFrame to the Excel file
				combined_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

		print(f"\n\nAll second-order GLDM features are extracted to the Excel file: {excel_file_name}")
		print("Please check the Excel file for further analysis and interpretation")

	def update_second_order_ngtdm_features_to_excel_file(
			self, excel_file_name, folders):
		excel_sheet_names = self.excel_sheet_names

		# Create an Excel writer object
		with pd.ExcelWriter(excel_file_name, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
			for index, folder in enumerate(folders):
				print(f"Extracting second-order NGTDM features from: {folder}")
				files = sorted(os.listdir(folder))
				last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
				features = Parallel(n_jobs=-1)(
					delayed(self.extract_second_order_gldm_features)(os.path.join(folder, file), file)
					for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
				)

				# with tqdm(total=len(files), desc=f"Folder: {last_two_folder_names}") as pbar:
				# 	features = Parallel(n_jobs=-1)(
				# 		delayed(self.extract_second_order_ngtdm_features)(os.path.join(folder, file), file)
				# 		for file in files
				# 	)
				# 	pbar.update()

				# Read the existing first order features from the Excel file
				first_order_features_df = pd.read_excel(excel_file_name, sheet_name=excel_sheet_names[index])

				# Create a new DataFrame with the second order GLCM features
				ngtdm_features_df = pd.DataFrame(features)

				# Merge the two DataFrames based on the 'Image Filename' column
				combined_features_df = pd.merge(first_order_features_df, ngtdm_features_df, on='Image Filename')

				# Write the combined DataFrame to the Excel file
				combined_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

		print(f"\n\nAll second-order NGTDM features are extracted to the Excel file: {excel_file_name}")
		print("Please check the Excel file for further analysis and interpretation")
