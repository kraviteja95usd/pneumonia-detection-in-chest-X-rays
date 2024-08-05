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

    def fetch_images_features_excel_file_path(self, excel_file_name):
        """
        Fetches the absolute path & name of the Excel file where the images info is saved.

        Args:
            excel_file_name(str): Takes in the name of the excel file to build absolute path

        Returns:
            Absolute path of the Excel file
        """

        excel_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), str(self.image_information_dir_name))
        excel_file = os.path.join(excel_file_path, str(excel_file_name))
        return excel_file

    @staticmethod
    def fetch_last_two_folder_names_of_path(folder_path):
        """
        Extract the last 2 folder names from a path

        Args:
            folder_path(str): Absolute path string of the folder path
            Example: /abc/def/ghi/jkl

        Returns:
            last_two_folder_names(str): Returns the last two folder names from the given path.
            Example from the above path: ghi/jkl
        """
        path_obj = pathlib.Path(folder_path)
        last_two_folder_names = f"{path_obj.parent.name}/{path_obj.name}"
        return last_two_folder_names

    @staticmethod
    def extract_first_order_features(image):
        """
        Calculate first-order features of an image.

        First-order features are statistics calculated from the pixel values of an image.
        They provide information about the distribution of pixel values, such as mean, median, standard deviation,
        variance, skewness, kurtosis, range, entropy, energy, uniformity, RMS, Maximum pixel intensity,
        Minimum pixel intensity, Median Absolute Deviation, Mean Absolute Deviation, and Interquartile range.

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

        # Calculate the Root Mean Square (RMS) of the image, measuring the average magnitude of pixel intensities.
        first_order_features["RMS"] = np.sqrt(np.mean(image ** 2))

        # Determine the maximum pixel intensity in the image.
        first_order_features["Max_Pixel_Intensity"] = np.max(image)

        # Determine the minimum pixel intensity in the image.
        first_order_features["Min_Pixel_Intensity"] = np.min(image)

        # Calculate the median of absolute deviations from the median, indicating the variability of pixel intensities.
        first_order_features["Median_Abs_Deviation"] = np.median(np.abs(image - np.median(image)))

        # Calculate the mean of absolute deviations from the mean, indicating the variability of pixel intensities.
        first_order_features["Mean_Abs_Deviation"] = np.mean(np.abs(image - np.mean(image)))

        # Calculate the Interquartile Range (IQR), measuring the spread of the middle 50% of pixel intensities.
        first_order_features["IQR"] = np.percentile(image, 75) - np.percentile(image, 25)

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
        glcm = graycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)

        # Extract basic GLCM properties
        # Measure of intensity contrast between a pixel and its neighbor over the whole image
        contrast = graycoprops(glcm, 'contrast')

        # Measure of how correlated a pixel is to its neighbor over the whole image
        correlation = graycoprops(glcm, 'correlation')

        # Measure of the sum of squared elements in the GLCM
        energy = graycoprops(glcm, 'energy')

        # Measure of how close the distribution of elements in the GLCM is to the GLCM diagonal
        homogeneity = graycoprops(glcm, 'homogeneity')

        # Calculate averages over all directions
        contrast = np.mean(contrast)
        correlation = np.mean(correlation)
        energy = np.mean(energy)
        homogeneity = np.mean(homogeneity)

        # Extract additional GLCM features
        glcm_flattened = glcm.flatten()

        # Dissimilarity: Measure of the dissimilarity between a pixel and its neighbor
        dissimilarity = np.mean(
            [
                np.sum(np.abs(row - col) * glcm[row, col]) for row in range(glcm.shape[0])
                for col in range(glcm.shape[1])
            ]
        )

        # Entropy: Measure of randomness in the GLCM
        glcm_flattened = glcm_flattened[glcm_flattened > 0]  # Remove zero entries to avoid log(0)
        entropy_value = entropy(glcm_flattened)

        # Auto-correlation: Measure of the correlation between a pixel and its neighbor
        row_indices, col_indices = np.indices(glcm.shape[:2])
        auto_correlation = np.mean([
            np.sum(row_indices * col_indices * glcm[:, :, 0, i]) for i in range(glcm.shape[3])
        ])

        # Cluster Prominence and Cluster Shade: Measures of the skewness and peakedness of the clusters in the GLCM
        mean_row = np.mean(row_indices)
        mean_col = np.mean(col_indices)
        cluster_prominence = np.mean([
            np.sum(((row_indices + col_indices - mean_row - mean_col) ** 4) * glcm[:, :, 0, i]) for i in
            range(glcm.shape[3])
        ])
        cluster_shade = np.mean([
            np.sum(((row_indices + col_indices - mean_row - mean_col) ** 3) * glcm[:, :, 0, i]) for i in
            range(glcm.shape[3])
        ])

        # Maximum Probability: Measure of the maximum probability in the GLCM
        max_probability = np.max(glcm)

        # Sum of Squares (Variance): Measure of the dispersion of elements in the GLCM
        row_indices, col_indices = np.indices(glcm.shape[:2])
        variances = []
        for i in range(glcm.shape[3]):
            sum_squares = np.sum((row_indices ** 2) * glcm[:, :, 0, i])
            variance = sum_squares / np.sum(glcm[:, :, 0, i])
            variances.append(variance)
        glcm_variance = np.mean(variances)

        # Sum Average, Sum Entropy, Sum Variance: Measures of the distribution of elements in the GLCM
        k_indices = np.arange(glcm.shape[0])
        p_x_plus_y = np.sum(glcm, axis=0)  # Sum over row

        # Normalize p_x_plus_y to ensure it sums up to 1
        p_x_plus_y = p_x_plus_y / np.sum(p_x_plus_y)

        sum_average = 0
        for i in range(glcm.shape[0]):
            sum_average += k_indices[i] * p_x_plus_y[i]

        sum_variance = 0
        for i in range(glcm.shape[0]):
            sum_variance += (k_indices[i] - sum_average) ** 2 * p_x_plus_y[i]

        epsilon = 1e-15  # small value to avoid division by zero
        p_x_plus_y_safe = np.where(p_x_plus_y > epsilon, p_x_plus_y, epsilon)
        sum_entropy = -np.sum(p_x_plus_y_safe * np.log2(p_x_plus_y_safe))

        sum_average = np.mean(sum_average)
        sum_variance = np.mean(sum_variance)

        # Difference Entropy and Difference Variance: Measures of the entropy and variance of the differences in the GLCM
        difference_sum = np.sum(glcm, axis=0)  # Sum over row and col
        difference_indices = np.arange(glcm.shape[0])
        difference_entropy = -np.sum([np.sum(difference_sum[:, i] * np.log(difference_sum[:, i] + 1e-10)) for i in
                                      range(difference_sum.shape[1])])
        difference_variance = np.sum((difference_indices[:, np.newaxis, np.newaxis] - np.sum(difference_sum, axis=0) /
                                      difference_sum.shape[0]) ** 2 * difference_sum)

        # Create a dictionary to store the extracted GLCM features
        glcm_features = {
            'GLCM_Contrast': contrast,
            'GLCM_Correlation': correlation,
            'GLCM_Energy': energy,
            'GLCM_Homogeneity': homogeneity,
            'GLCM_Dissimilarity': dissimilarity,
            'GLCM_Entropy': entropy_value,
            'GLCM_Auto_correlation': auto_correlation,
            'GLCM_Cluster_Prominence': cluster_prominence,
            'GLCM_Cluster_Shade': cluster_shade,
            'GLCM_Max_Probability': max_probability,
            'GLCM_Variance': glcm_variance,
            'GLCM_Sum_Average': sum_average,
            'GLCM_Sum_Entropy': sum_entropy,
            'GLCM_Sum_Variance': sum_variance,
            'GLCM_Difference_Entropy': difference_entropy,
            'GLCM_Difference_Variance': difference_variance
        }

        # Return the dictionary containing the extracted GLCM features
        return glcm_features

    # Extract GLRLM Features from an image
    @staticmethod
    def extract_glrlm_features(image):

        """
        Extracts GLRLM (Gray Level Run Length Matrix) features from the given image.

        Args:
            image (numpy.ndarray): The input image (numpy array representing the input image as an argument).

        Returns:
            glrlm_features(dict): A dictionary containing the extracted GLRLM features.
        """
        # Evaluate GLRLM
        # Get the maximum gray level in the image
        max_gray_level = np.max(image)

        # Initialize the GLRLM matrix with zeros
        glrlm = np.zeros((max_gray_level + 1, image.shape[0] + image.shape[1] - 1))

        # Horizontal Runs
        for row_index in range(image.shape[0]):
            column_index = 0
            while column_index < image.shape[1]:
                gray_level = image[row_index, column_index]
                run_length = 1
                while column_index + 1 < image.shape[1] and image[row_index, column_index + 1] == gray_level:
                    column_index += 1
                    run_length += 1
                glrlm[gray_level, run_length - 1] += 1
                column_index += 1

        # Vertical Runs
        for column_index in range(image.shape[1]):
            row_index = 0
            while row_index < image.shape[0]:
                gray_level = image[row_index, column_index]
                run_length = 1
                while row_index + 1 < image.shape[0] and image[row_index + 1, column_index] == gray_level:
                    row_index += 1
                    run_length += 1
                glrlm[gray_level, run_length - 1] += 1
                row_index += 1

        # Calculate the number of runs
        number_of_runs = np.sum(glrlm)

        # Calculate the number of pixels in the image
        number_of_pixels = image.shape[0] * image.shape[1]

        # Calculate the mean run length
        mean_run_length = np.sum(glrlm * np.arange(1, glrlm.shape[1] + 1)) / number_of_runs

        # Short Run Emphasis (SRE)
        short_run_emphasis = np.sum(glrlm[:, 0:2]) / number_of_runs

        # Long Run Emphasis (LRE)
        # long_run_emphasis = np.sum(glrlm[:, -2:]) / number_of_runs
        run_lengths = np.arange(1, glrlm.shape[1] + 1)  # Run lengths are 1, 2, 3, ..., max run length
        long_run_emphasis = np.sum(glrlm * (run_lengths ** 2)) / number_of_runs

        # Gray Level Non-Uniformity (GLN)
        gray_level_non_uniformity = np.sum(np.sum(glrlm, axis=1) ** 2) / number_of_runs

        # Run Length Non-Uniformity (RLN)
        run_length_non_uniformity = np.sum(np.sum(glrlm, axis=0) ** 2) / number_of_runs

        # Run Percentage (RP)
        run_percentage = number_of_runs / number_of_pixels

        # Low Gray Level Run Emphasis (LGLRE)
        low_gray_level_run_emphasis = np.sum(
            glrlm / (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # High Gray Level Run Emphasis (HGLRE)
        high_gray_level_run_emphasis = np.sum(
            glrlm * (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # Short Run Low Gray Level Emphasis (SRLGLE)
        short_run_low_gray_level_emphasis = np.sum(
            glrlm[:, 0:2] / (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # Short Run High Gray Level Emphasis (SRHGLE)
        short_run_high_gray_level_emphasis = np.sum(
            glrlm[:, 0:2] * (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # Long Run Low Gray Level Emphasis (LRLGLE)
        # long_run_low_gray_level_emphasis = np.sum(
        #     glrlm[:, -2:] / (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # Long Run Low Gray Level Emphasis (LRLGLE)
        long_run_low_gray_level_emphasis = np.sum(
            glrlm[:, 1:] / (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # Long Run High Gray Level Emphasis (LRHGLE)
        # long_run_high_gray_level_emphasis = np.sum(
        #     glrlm[:, -2:] * (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # Long Run High Gray Level Emphasis (LRHGLE)
        long_run_high_gray_level_emphasis = np.sum(
            glrlm[:, 1:] * (np.arange(1, max_gray_level + 2).reshape(-1, 1) ** 2)) / number_of_runs

        # Run Variance (RV)
        run_length_indices = np.arange(1, glrlm.shape[1] + 1)
        run_variance = np.sum(glrlm * (run_length_indices - mean_run_length) ** 2) / number_of_runs

        # Run Entropy (RE)
        run_probabilities = glrlm / number_of_runs
        run_entropy = -np.sum(run_probabilities * np.log2(run_probabilities + np.finfo(float).eps))

        # Difference Average (DA)
        gray_level_diff = np.abs(
            np.arange(max_gray_level + 1).reshape(-1, 1) - np.arange(max_gray_level + 1).reshape(1, -1))
        glrlm_sum = np.sum(glrlm, axis=1, keepdims=True)  # Sum along the run length axis
        glrlm_sum = np.broadcast_to(glrlm_sum,
                                    (max_gray_level + 1,
                                     max_gray_level + 1))  # Broadcast to match gray_level_diff's shape
        difference_average = np.sum(glrlm_sum * gray_level_diff) / (number_of_runs * (max_gray_level + 1))

        # Difference Variance (DV)
        difference_variance = np.sum(glrlm_sum * (gray_level_diff - difference_average) ** 2) / (
                number_of_runs * (max_gray_level + 1))

        # Difference Entropy (DE)
        difference_probabilities = gray_level_diff / (number_of_runs * (max_gray_level + 1))
        difference_entropy = -np.sum(difference_probabilities * np.log2(difference_probabilities + np.finfo(float).eps))

        # Create a dictionary to store the extracted GLRLM features
        glrlm_features = {
            'GLRLM_ShortRunEmphasis': short_run_emphasis,
            'GLRLM_LongRunEmphasis': long_run_emphasis,
            'GLRLM_GrayLevelNon_Uniformity': gray_level_non_uniformity,
            'GLRLM_RunLengthNon_Uniformity': run_length_non_uniformity,
            'GLRLM_RunPercentage': run_percentage,
            'GLRLM_LowGrayLevelRunEmphasis': low_gray_level_run_emphasis,
            'GLRLM_HighGrayLevelRunEmphasis': high_gray_level_run_emphasis,
            'GLRLM_ShortRunLowGrayLevelEmphasis': short_run_low_gray_level_emphasis,
            'GLRLM_ShortRunHighGrayLevelEmphasis': short_run_high_gray_level_emphasis,
            'GLRLM_LongRunLowGrayLevelEmphasis': long_run_low_gray_level_emphasis,
            'GLRLM_LongRunHighGrayLevelEmphasis': long_run_high_gray_level_emphasis,
            'GLRLM_RunVariance': run_variance,
            'GLRLM_RunEntropy': run_entropy,
            'GLRLM_DifferenceAverage': difference_average,
            'GLRLM_DifferenceVariance': difference_variance,
            'GLRLM_DifferenceEntropy': difference_entropy,
            "Num_of_runs": number_of_runs,
            "Num_of_pixels": number_of_pixels
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
            gldm_features(dict): A dictionary containing the extracted GLDM features.
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

        # Calculate the small dependence emphasis (SDE)
        small_dependence_emphasis = np.sum(gldm[0:2, :]) / np.sum(gldm)

        # Calculate the large dependence emphasis (LDE)
        large_dependence_emphasis = np.sum(gldm[-2:, :]) / np.sum(gldm)

        # Calculate the gray level non-uniformity (GLN)
        gray_level_non_uniformity = np.sum(np.sum(gldm, axis=1) ** 2) / np.sum(gldm)

        # Calculate the dependence count non-uniformity (DCN)
        dependence_count_non_uniformity = np.sum(np.sum(gldm, axis=0) ** 2) / np.sum(gldm)

        # Calculate the dependence count entropy
        dependence_count_entropy = -np.sum(gldm * np.log(gldm + np.finfo(float).eps))

        # Calculate the gray level entropy
        gray_level_entropy = -np.sum(gldm * np.log(gldm + np.finfo(float).eps))

        # Calculate the dependence count mean
        dependence_count_mean = np.mean(np.sum(gldm, axis=0))

        # Calculate the gray level mean
        gray_level_mean = np.mean(np.sum(gldm, axis=1))

        # Calculate the dependence count variance
        dependence_count_variance = np.var(np.sum(gldm, axis=0))

        # Calculate the gray level variance
        gray_level_variance = np.var(np.sum(gldm, axis=1))

        # Calculate the dependence count energy
        dependence_count_energy = np.sum(gldm ** 2)

        # Calculate the gray level energy
        gray_level_energy = np.sum(gldm ** 2)

        # Calculate the dependence count maximum
        dependence_count_maximum = np.max(np.sum(gldm, axis=0))

        # Calculate the gray level maximum
        gray_level_maximum = np.max(np.sum(gldm, axis=1))

        # Calculate the dependence count contrast
        dependence_count_contrast = np.sum(np.abs(np.arange(max_gray_level + 1)[:, None] - np.arange(max_gray_level + 1)) * gldm)

        # Calculate the gray level contrast
        gray_level_contrast = np.sum(np.abs(np.arange(max_gray_level + 1)[:, None] - np.arange(max_gray_level + 1)) * gldm)

        # Calculate the dependence count correlation
        dependence_count_correlation = np.corrcoef(np.sum(gldm, axis=0))

        # Calculate the gray level correlation
        gray_level_correlation = np.corrcoef(np.sum(gldm, axis=1))

        # Calculate the dependence count homogeneity
        dependence_count_homogeneity = np.sum(gldm / (1.0 + np.abs(np.arange(max_gray_level + 1)[:, None] - np.arange(max_gray_level + 1))))

        # Calculate the gray level homogeneity
        gray_level_homogeneity = np.sum(gldm / (1.0 + np.abs(np.arange(max_gray_level + 1)[:, None] - np.arange(max_gray_level + 1))))

        # Calculate the dependence count sum
        dependence_count_sum = np.sum(np.sum(gldm, axis=0))

        # Calculate the gray level sum
        gray_level_sum = np.sum(np.sum(gldm, axis=1))

        # Calculate the dependence count range
        dependence_count_range = np.max(np.sum(gldm, axis=0)) - np.min(np.sum(gldm, axis=0))

        # Calculate the gray level range
        gray_level_range = np.max(np.sum(gldm, axis=1)) - np.min(np.sum(gldm, axis=1))

        # Create a dictionary to store the extracted GLDM features
        gldm_features = {
            'GLDM_SmallDependenceEmphasis': small_dependence_emphasis,
            'GLDM_LargeDependenceEmphasis': large_dependence_emphasis,
            'GLDM_GrayLevelNon_Uniformity': gray_level_non_uniformity,
            'GLDM_DependenceCountNon_Uniformity': dependence_count_non_uniformity,
            'GLDM_DependenceCountEntropy': dependence_count_entropy,
            'GLDM_GrayLevelEntropy': gray_level_entropy,
            'GLDM_DependenceCountMean': dependence_count_mean,
            'GLDM_GrayLevelMean': gray_level_mean,
            'GLDM_DependenceCountVariance': dependence_count_variance,
            'GLDM_GrayLevelVariance': gray_level_variance,
            'GLDM_DependenceCountEnergy': dependence_count_energy,
            'GLDM_GrayLevelEnergy': gray_level_energy,
            'GLDM_DependenceCountMaximum': dependence_count_maximum,
            'GLDM_GrayLevelMaximum': gray_level_maximum,
            'GLDM_DependenceCountContrast': dependence_count_contrast,
            'GLDM_GrayLevelContrast': gray_level_contrast,
            'GLDM_DependenceCountCorrelation': dependence_count_correlation,
            'GLDM_GrayLevelCorrelation': gray_level_correlation,
            'GLDM_DependenceCountHomogeneity': dependence_count_homogeneity,
            'GLDM_GrayLevelHomogeneity': gray_level_homogeneity,
            'GLDM_DependenceCountSum': dependence_count_sum,
            'GLDM_GrayLevelSum': gray_level_sum,
            'GLDM_DependenceCountRange': dependence_count_range,
            'GLDM_GrayLevelRange': gray_level_range
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
            ngtdm_features(dict): A dictionary containing the extracted NGTDM features.
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

        # Calculate the dissimilarity feature
        # Dissimilarity is the sum of the absolute differences between the NGTDM matrix and the mean NGTDM value
        dissimilarity = np.sum(np.abs(ngtdm - np.mean(ngtdm))) / (image.shape[0] * image.shape[1])

        # Calculate the joint energy feature
        # Joint energy is the sum of the squared NGTDM matrix divided by the total number of pixels in the image
        joint_energy = np.sum(ngtdm * ngtdm) / (image.shape[0] * image.shape[1])

        # Calculate the joint entropy feature
        # Joint entropy is the sum of the product of the NGTDM matrix and the logarithm of the NGTDM matrix plus 1,
        # divided by the total number of pixels in the image
        joint_entropy = np.sum(ngtdm * np.log2(ngtdm + 1)) / (image.shape[0] * image.shape[1])

        # Calculate the informational measure of correlation (IMC) 1 feature
        # IMC 1 is the sum of the product of the NGTDM matrix and the logarithm of the NGTDM matrix plus 1,
        # divided by the total number of pixels in the image
        informational_measure_of_correlation1 = np.sum(ngtdm * np.log2(ngtdm + 1)) / (image.shape[0] * image.shape[1])

        # Calculate the inverse difference moment (IDM) feature
        # IDM is the sum of the NGTDM matrix divided by the sum of the absolute differences between the image pixels
        # and the mean image pixel value
        inverse_diff_moment = np.sum(ngtdm) / np.sum(np.abs(image - np.mean(image)))

        # Calculate the inverse difference moment normalized (IDMN) feature
        # IDMN is the sum of the NGTDM matrix divided by the sum of the absolute differences between the image pixels
        # and the mean image pixel value,
        # normalized by the maximum gray level
        inverse_diff_moment_norm = np.sum(ngtdm) / (np.sum(np.abs(image - np.mean(image))) * max_gray_level)

        # Calculate the inverse difference (ID) feature
        # ID is the sum of the NGTDM matrix divided by the sum of the absolute differences between the image pixels
        # and the mean image pixel value
        inverse_difference = np.sum(ngtdm) / np.sum(np.abs(image - np.mean(image)))

        # Calculate the inverse variance feature
        # Inverse variance is the sum of the NGTDM matrix divided by the variance of the image
        inverse_variance = np.sum(ngtdm) / np.var(image)

        # Calculate the maximum probability feature
        # Maximum probability is the maximum value in the NGTDM matrix
        max_prob = np.max(ngtdm)

        # Calculate the sum average feature
        # Sum average is the sum of the NGTDM matrix divided by the total number of pixels in the image
        sum_avg = np.sum(ngtdm) / (image.shape[0] * image.shape[1])

        # Calculate the sum entropy feature
        # Sum entropy is the sum of the product of the NGTDM matrix and the logarithm of the NGTDM matrix plus 1,
        # divided by the total number of pixels in the image
        sum_entropy = np.sum(ngtdm * np.log2(ngtdm + 1)) / (image.shape[0] * image.shape[1])

        # Calculate the sum of squares feature
        # Sum of squares is the sum of the squared NGTDM matrix divided by the total number of pixels in the image
        sum_squares = np.sum(ngtdm * ngtdm) / (image.shape[0] * image.shape[1])

        # Calculate the auto-correlation feature
        # Auto-correlation is the sum of the product of the NGTDM matrix and the NGTDM matrix shifted by one position,
        # divided by the total number of pixels in the image
        auto_correlation = np.sum(ngtdm * np.roll(ngtdm, 1)) / (image.shape[0] * image.shape[1])

        # Calculate the cluster prominence feature
        # Cluster prominence is the sum of the squared NGTDM matrix divided by the sum of the NGTDM matrix
        cluster_prominence = np.sum(ngtdm * ngtdm) / np.sum(ngtdm)

        # Calculate the cluster shade feature
        # Cluster shade is the sum of the cubed NGTDM matrix divided by the sum of the NGTDM matrix
        cluster_shade = np.sum(ngtdm * ngtdm * ngtdm) / np.sum(ngtdm)

        # Calculate the cluster tendency feature
        # Cluster tendency is the sum of the fourth power of the NGTDM matrix divided by the sum of the NGTDM matrix
        cluster_tendency = np.sum(ngtdm * ngtdm * ngtdm * ngtdm) / np.sum(ngtdm)

        # Calculate the correlation feature
        # Correlation is the sum of the product of the NGTDM matrix and the NGTDM matrix shifted by one position,
        # divided by the total number of pixels in the image
        correlation = np.sum(ngtdm * np.roll(ngtdm, 1)) / (image.shape[0] * image.shape[1])

        # Calculate the variance feature
        # Variance is the sum of the squared differences between the NGTDM matrix and the mean NGTDM value
        variance = np.sum((ngtdm - np.mean(ngtdm)) ** 2)

        # Create a dictionary to store the extracted NGTDM features
        ngtdm_features = {
            'NGTDM_Coarseness': coarseness,
            'NGTDM_Contrast': contrast,
            'NGTDM_Busyness': busyness,
            'NGTDM_Complexity': complexity,
            'NGTDM_Dissimilarity': dissimilarity,
            'NGTDM_JointEnergy': joint_energy,
            'NGTDM_JointEntropy': joint_entropy,
            'NGTDM_Informational_Measure_of_Correlation1': informational_measure_of_correlation1,
            'NGTDM_Inverse_Difference_Moment': inverse_diff_moment,
            'NGTDM_Inverse_Difference_Moment_Normmalized': inverse_diff_moment_norm,
            'NGTDM_Inverse_Difference': inverse_difference,
            'NGTDM_InverseVariance': inverse_variance,
            'NGTDM_MaxProb': max_prob,
            'NGTDM_SumAvg': sum_avg,
            'NGTDM_SumEntropy': sum_entropy,
            'NGTDM_SumSquares': sum_squares,
            'NGTDM_Autocorrelation': auto_correlation,
            'NGTDM_ClusterProminence': cluster_prominence,
            'NGTDM_ClusterShade': cluster_shade,
            'NGTDM_ClusterTendency': cluster_tendency,
            'NGTDM_Correlation': correlation,
            'NGTDM_Variance': variance
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

    def update_first_order_features_to_excel_file(self, folders):
        """
        Extract the first order features and write to an Excel file.

        Args:
            folders(list): List of folders where the train (NORMAL & PNEUMONIA) and test (NORMAL & PNEUMONIA) images are present.

        Returns:
            None

        Prints:
            Progress bars that display the status of feature extraction of first order features and  path to the Excel file.
        """

        excel_sheet_names = self.excel_sheet_names
        excel_file_name = self.fetch_images_features_excel_file_path(self.image_first_order_features_xls_file_name)
        print(f"Extracted First order features will be saved - {excel_file_name}\n\n")

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

    def update_second_order_glcm_features_to_excel_file(self, folders):
        """
        Extract the second order GLCM features and write to an Excel file.

        Args:
            folders(list): List of folders where the train (NORMAL & PNEUMONIA) and test (NORMAL & PNEUMONIA) images are present.

        Returns:
            None

        Prints:
            Progress bars that display the status of feature extraction of second order GLCM features and  path to the Excel file.
        """

        excel_sheet_names = self.excel_sheet_names
        excel_file_name = self.fetch_images_features_excel_file_path(
            self.image_second_order_features_glcm_xls_file_name)
        print(f"Extracted GLCM features will be saved to - {excel_file_name}\n\n")

        # Create an Excel writer object
        with pd.ExcelWriter(excel_file_name) as writer:
            for index, folder in enumerate(folders):
                print(f"Extracting second-order features GLCM from: {folder}")
                files = sorted(os.listdir(folder))
                last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)

                features = Parallel(n_jobs=-1)(
                    delayed(self.extract_second_order_glcm_features)(os.path.join(folder, file), file)
                    for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
                )

                glcm_features_df = pd.DataFrame(features)
                glcm_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

        print(f"\n\nAll first-order features are extracted to the Excel file: {excel_file_name}")
        print("Please check the Excel file for further analysis and interpretation")

    def update_second_order_glrlm_features_to_excel_file(self, folders):
        """
        Extract the second order GLRLM features and write to an Excel file.

        Args:
            folders(list): List of folders where the train (NORMAL & PNEUMONIA) and test (NORMAL & PNEUMONIA) images are present.

        Returns:
            None

        Prints:
            Progress bars that display the status of feature extraction of second order GLRLM features and  path to the Excel file.
        """

        excel_sheet_names = self.excel_sheet_names
        excel_file_name = self.fetch_images_features_excel_file_path(
            self.image_second_order_features_glrlm_xls_file_name)
        print(f"Extracted GLRLM features will be saved to - {excel_file_name}\n\n")

        # Create an Excel writer object
        with pd.ExcelWriter(excel_file_name) as writer:
            for index, folder in enumerate(folders):
                print(f"Extracting second-order features GLRLM from: {folder}")
                files = sorted(os.listdir(folder))
                last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
                features = Parallel(n_jobs=-1)(
                    delayed(self.extract_second_order_glrlm_features)(os.path.join(folder, file), file)
                    for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
                )
                extracted_glrlm_features_df = pd.DataFrame(features)
                extracted_glrlm_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

        print(f"\n\nAll first-order features are extracted to the Excel file: {excel_file_name}")
        print("Please check the Excel file for further analysis and interpretation")

    def update_second_order_gldm_features_to_excel_file(self, folders):
        """
        Extract the second order GLDM features and write to an Excel file.

        Args:
            folders(list): List of folders where the train (NORMAL & PNEUMONIA) and test (NORMAL & PNEUMONIA) images are present.

        Returns:
            None

        Prints:
            Progress bars that display the status of feature extraction of second order GLDM features and  path to the Excel file.
        """

        excel_sheet_names = self.excel_sheet_names
        excel_file_name = self.fetch_images_features_excel_file_path(
            self.image_second_order_features_gldm_xls_file_name)
        print(f"Extracted GLDM features will be saved to - {excel_file_name}\n\n")

        # Create an Excel writer object
        with pd.ExcelWriter(excel_file_name) as writer:
            for index, folder in enumerate(folders):
                print(f"Extracting second-order features GLDM from: {folder}")
                files = sorted(os.listdir(folder))
                last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
                features = Parallel(n_jobs=-1)(
                    delayed(self.extract_second_order_gldm_features)(os.path.join(folder, file), file)
                    for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
                )
                extracted_gldm_features_df = pd.DataFrame(features)
                extracted_gldm_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

        print(f"\n\nAll first-order features are extracted to the Excel file: {excel_file_name}")
        print("Please check the Excel file for further analysis and interpretation")

    def update_second_order_ngtdm_features_to_excel_file(self, folders):
        """
        Extract the second order NGTDM features and write to an Excel file.

        Args:
            folders(list): List of folders where the train (NORMAL & PNEUMONIA) and test (NORMAL & PNEUMONIA) images are present.

        Returns:
            None

        Prints:
            Progress bars that display the status of feature extraction of second order NGTDM features and  path to the Excel file.
        """

        excel_sheet_names = self.excel_sheet_names
        excel_file_name = self.fetch_images_features_excel_file_path(
            self.image_second_order_features_ngtdm_xls_file_name)
        print(f"Extracted NGTDM features will be saved to - {excel_file_name}\n\n")

        # Create an Excel writer object
        with pd.ExcelWriter(excel_file_name) as writer:
            for index, folder in enumerate(folders):
                print(f"Extracting second-order features NGTDM from: {folder}")
                files = sorted(os.listdir(folder))
                last_two_folder_names = self.fetch_last_two_folder_names_of_path(folder)
                features = Parallel(n_jobs=-1)(
                    delayed(self.extract_second_order_ngtdm_features)(os.path.join(folder, file), file)
                    for file in tqdm(files, desc=f"Folder: {last_two_folder_names}")
                )
                extracted_ngtdm_features_df = pd.DataFrame(features)
                extracted_ngtdm_features_df.to_excel(writer, sheet_name=excel_sheet_names[index], index=False)

        print(f"\n\nAll first-order features are extracted to the Excel file: {excel_file_name}")
        print("Please check the Excel file for further analysis and interpretation")
