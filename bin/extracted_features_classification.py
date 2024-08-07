from os import getcwd
from sys import path

path.append(getcwd() + "/bin")

import itertools
import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import ssl
from openpyxl import load_workbook

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

from pycaret.classification import setup, compare_models, evaluate_model, get_config
from image_pre_processing import ImageDataPreprocessing
from pneumonia_detector_constants import pneumonia_detector_constants


class ExtractedFeaturesClassification(ImageDataPreprocessing):
    def __init__(self):
        """
            Initialize ExtractedFeaturesClassification class by inheriting the properties of ImageDataPreprocessing class
        """
        super().__init__()
        self.target_column_name = pneumonia_detector_constants["target_column_name"]
        self.trained_model_pkl_files_dir_name = pneumonia_detector_constants["trained_model_pkl_files_dir_name"]
        self.excel_sheet_names = pneumonia_detector_constants["excel_sheet_names"]
        self.image_first_order_features_xls_file_name = pneumonia_detector_constants["image_first_order_features_xls_file_name"]
        self.image_second_order_features_glcm_xls_file_name = pneumonia_detector_constants["image_second_order_features_glcm_xls_file_name"]
        self.image_second_order_features_glrlm_xls_file_name = pneumonia_detector_constants["image_second_order_features_glrlm_xls_file_name"]
        self.image_second_order_features_gldm_xls_file_name = pneumonia_detector_constants["image_second_order_features_gldm_xls_file_name"]
        self.image_first_order_features_xls_file_name = pneumonia_detector_constants["image_second_order_features_ngtdm_xls_file_name"]

    def load_excel_file_into_dataframe(self, excel_file_name, excel_sheet_name):
        """
        Fetches the absolute path & name of the Excel file where the images info is saved.

        Args:
            excel_file_name(str): Name of the Excel file to be loaded to a dataframe
            excel_sheet_name(str): Name of the Excel sheet inside the file to be loaded to a dataframe

        Returns:
            Absolute path of the Excel file
        """
        excel_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), str(self.image_information_dir_name))
        excel_file_abs_path = os.path.join(excel_file_path, str(excel_file_name))

        excel_file_df = pd.read_excel(excel_file_abs_path, sheet_name=excel_sheet_name)

        return excel_file_df

    def add_target_column_to_dataframes(self, df_train_normal, df_train_pneumonia, df_test_normal, df_test_pneumonia):
        """
        Fetches the absolute path & name of the Excel file where the images info is saved.

        Args:
            df_train_normal(pd.DataFrame): Train Normal Dataframe
            df_train_pneumonia(pd.DataFrame): Train Pneumonia Dataframe
            df_test_normal(pd.DataFrame): Test Normal Dataframe
            df_test_pneumonia(pd.DataFrame): Test Pneumonia Dataframe

        Returns:
            df_train_normal(pd.DataFrame): Train Normal Dataframe after adding target column to it
            df_train_pneumonia(pd.DataFrame): Train Pneumonia Dataframe after adding target column to it
            df_test_normal(pd.DataFrame): Test Normal Dataframe after adding target column to it
            df_test_pneumonia(pd.DataFrame): Test Pneumonia Dataframe after adding target column to it
        """
        df_train_normal[self.target_column_name] = 0
        df_train_pneumonia[self.target_column_name] = 1
        df_test_normal[self.target_column_name] = 0
        df_test_pneumonia[self.target_column_name] = 1

        return df_train_normal, df_train_pneumonia, df_test_normal, df_test_pneumonia

    @staticmethod
    def concatenate_dataframes(dataframe1, dataframe2):
        """
        Fetches the absolute path & name of the Excel file where the images info is saved.

        Args:
            dataframe1(pd.DataFrame): First dataframe
            dataframe2(pd.DataFrame): Second dataframe to be concatenated with first dataframe

        Returns:
            final_dataframe(pd.DataFrame): Concatenated dataframe
        """
        final_dataframe = pd.concat([dataframe1, dataframe2], ignore_index=True)
        return final_dataframe

    def train_multiple_ml_models(self, train_df, test_df):
        """
        Train baseline models with provided dataframes using all the ML algorithms
        Executes the required steps to compare multiple models, print classification report and
        returns the best ML model for the passed dataframes.

        Args:
            train_df(pd.DataFrame): Dataframe used to train the model
            test_df(pd.DataFrame): Dataframe used to test the model

        Returns:
            best_model(algorithm type): Returns the comparison of multiple ML models along with the best model
            Example: sklearn.linear_model._logistic.LogisticRegression
        """

        setup(train_df, test_data=test_df, target=self.target_column_name, session_id=123, index=False, verbose=True)
        best_model = compare_models()
        return best_model

    @staticmethod
    def evaluate_best_model(best_model):
        """
        Evaluates the absolute path & name of the Excel file where the images info is saved.

        Args:
            best_model(pd.DataFrame): Dataframe used to train the model

        Returns:
            evaluated model of the best_model in interactive mode
        """
        return evaluate_model(best_model)

    @staticmethod
    def check_missing_values_of_dataframe(dataframe: pd.DataFrame) -> pd.Series:
        """
        Returns the count of null values for each column in the input dataframe.

        Args:
            dataframe(pd.DataFrame): The input dataframe.

        Returns:
            pd.Series: The count of null values for each column.
        """
        return dataframe.isna().sum()

    @staticmethod
    def check_duplicates_of_dataframe(dataframe: pd.DataFrame) -> int:
        """
        Returns the number of duplicate rows in the input dataframe.

        Args:
            dataframe (pd.DataFrame): The input dataframe.

        Returns:
            int: The number of duplicate rows.
        """
        return dataframe.duplicated().sum()

    @staticmethod
    def plots_for_all_features(dataframe, features):
        num_rows = len(features)

        # Create subplots: num_rows rows and 2 columns
        _, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 6))

        colors = itertools.cycle(['skyblue', 'lightgreen', 'magenta', 'red', 'orange', 'yellow'])

        for i, feature in enumerate(features):
            # Plot the Violin Plot in the first column
            sns.violinplot(data=[dataframe[feature]], ax=axes[i, 0], color=next(colors))
            axes[i, 0].set_title(f'Violin Plot for {feature}')

            # Plot Histogram with KDE in the second column
            sns.histplot(data=dataframe[feature], kde=True, ax=axes[i, 1], color=next(colors))
            axes[i, 1].set_title(f'Histogram for {feature}')

        # Adjust layout within window
        plt.tight_layout()
        plt.show()

    @staticmethod
    def return_pre_processed_dataframes_from_setup_method():
        x_train = get_config('X_train')  # Preprocessed training features
        y_train = get_config('y_train')  # Preprocessed training target
        x_test = get_config('X_test')  # Preprocessed testing features
        y_test = get_config('y_test')  # Preprocessed testing target

        # Combine features and targets into DataFrames for better readability
        preprocessed_train_df = pd.concat([x_train, y_train], axis=1)
        preprocessed_test_df = pd.concat([x_test, y_test], axis=1)

        return preprocessed_train_df, preprocessed_test_df

    def save_trained_model(self, final_model, model_file_name):
        """
        Save the trained model as a pickle file in the required directory

        Parameters:
            final_model(pycaret.Pipeline): Trained and finalized ML model that needs to be saved in directory
            model_file_name(str): Name of the model pickle file to be saved

        Returns:
            None
        """
        # Define the directory path and custom name
        directory_path = self.trained_model_pkl_files_dir_name

        # Combine the directory path and custom name
        model_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), str(directory_path))
        model_file_full_path = os.path.join(model_file_path, model_file_name)

        # Create directory if it does not exist
        if not os.path.exists(model_file_path):
            os.makedirs(model_file_path)

        print("Model is saved to: ", joblib.dump(final_model, model_file_full_path))
