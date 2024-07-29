import os


def count_images(dataset_path):
	"""
	Capture the count of images in each folder/subfolder

	Args:
		 dataset_path(str): Absolute path of the image dataset directory from the get_base_path_of_dataset function

	Returns:
		Prints the count of train/test images in each folder/subfolder
	"""
	# current_dir = os.path.dirname(__file__)
	# print("Current directory:", current_dir)
	#
	# dataset_folder = os.path.join(current_dir, 'dataset')
	# dataset_folder_name = os.path.join(dataset_folder, 'chest_xray')
	# print("Dataset folder name:", dataset_folder_name)

	train_normal = os.path.join(dataset_path, "train", "NORMAL")
	train_pneumonia = os.path.join(dataset_path, "train", "PNEUMONIA")
	test_normal = os.path.join(dataset_path, "test", "NORMAL")
	test_pneumonia = os.path.join(dataset_path, "test", "PNEUMONIA")

	train_normal_count = len(
		[name for name in os.listdir(train_normal) if os.path.isfile(os.path.join(train_normal, name))])
	train_pneumonia_count = len(
		[name for name in os.listdir(train_pneumonia) if os.path.isfile(os.path.join(train_pneumonia, name))])
	test_normal_count = len(
		[name for name in os.listdir(test_normal) if os.path.isfile(os.path.join(test_normal, name))])
	test_pneumonia_count = len(
		[name for name in os.listdir(test_pneumonia) if os.path.isfile(os.path.join(test_pneumonia, name))])

	print(train_normal_count)
	print(train_pneumonia_count)
	print(test_normal_count)
	print(test_pneumonia_count)


dataset_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
# print(dataset_folder)
dataset_folder_name = os.path.join(dataset_folder, 'chest_xray')
# print(dataset_folder_name)
count_images(dataset_folder_name)
