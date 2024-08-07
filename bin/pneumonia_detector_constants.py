pneumonia_detector_constants = {
	"chest_xray_folder_name": "chest_xray",
	"dataset_dir_name": "dataset",
	"image_information_dir_name": "image_information",
	"trained_model_pkl_files_dir_name": "trained_models",
	"image_info_xls_file_name": "chest_xray_images_info.xlsx",
	"image_first_order_features_xls_file_name": "chest_xray_images_first_order_features.xlsx",
	"image_second_order_features_glcm_xls_file_name": "chest_xray_images_second_order_features_glcm.xlsx",
	"image_second_order_features_glrlm_xls_file_name": "chest_xray_images_second_order_features_glrlm.xlsx",
	"image_second_order_features_gldm_xls_file_name": "chest_xray_images_second_order_features_gldm.xlsx",
	"image_second_order_features_ngtdm_xls_file_name": "chest_xray_images_second_order_features_ngtdm.xlsx",
	"images_features_merged_xls_file_name": "images_features_merged.xlsx",
	"train_test_image_dirs": [
		"train", "test"
	],
	"normal_pneumonia_image_dirs": [
		"NORMAL", "PNEUMONIA"
	],
	"excel_sheet_names": [
			'Train Normal Features', 'Train Pneumonia Features',
			'Test Normal Features', 'Test Pneumonia Features'
		],
	"target_column_name": "Pneumonia"
}