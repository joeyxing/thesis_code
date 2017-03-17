#!/home/joey/Apps/TensorFlow/bin/python3
import os
import pickle
import random
import sys
from shutil import move, copy2

# import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

IMAGE_DIR = '/home/joey/Work/thesis_code/data/Nodule'
IMAGE_SIZE = 100
IMAGE_DEPTH = 255
CHANNEL_NUM = 3
PROJECT_ROOT = '/home/joey/Work/thesis_code'

MALI_LABEL_DICT = {}
with open('/home/joey/Work/thesis_code/labels.txt', 'r+') as label_file:
    labels = label_file.read().split('\n')
    labels.pop(-1)
    for filename_label in labels:
        tmp_file_label = filename_label.split(' ')
        # is_mali = False
        # if tmp_file_label[1] == 'true':
            # is_mali = True
        MALI_LABEL_DICT[tmp_file_label[0]] = int(tmp_file_label[1]) - 1

# Data Preparation #
# load label file #
def rmdir_(dir_name):
    children = os.listdir(dir_name)
    for x in children:
        child = os.path.join(dir_name, x)
        if os.path.isdir(child):
            rmdir_(child)
        elif os.path.isfile(child):
            os.remove(child)
    os.rmdir(dir_name)


def separate_BM(from_data=IMAGE_DIR, force=False):
    """
    """
    print(from_data)

    if force:
        for i in range(5):
            try:
                print("Try removing directory %d" % i)
                rmdir_(os.path.join(PROJECT_ROOT, str(i)))
            except FileNotFoundError as err:
                pass

    else:
        status_ok = True
        for i in range(5):
            status_ok = status_ok and os.path.isdir(os.path.join(PROJECT_ROOT, str(i)))
        if status_ok:
            return [os.listdir(os.path.join(PROJECT_ROOT, str(x))) for x in range(5)]
        else:
            print("Current data broken, use force=True")
            return None


    for i in range(5):
        os.mkdir(os.path.join(PROJECT_ROOT, str(i)))
    for image in os.listdir(from_data):
        class_dir = os.path.join(PROJECT_ROOT, str(MALI_LABEL_DICT[image]))
        copy2(os.path.join(from_data, image), class_dir)
    return [os.listdir(os.path.join(PROJECT_ROOT, str(x))) for x in range(5)]


    

mali_classes_images = separate_BM(force=True)


TEST_PER_CLASS = 40
VALID_PER_CLASS = 40
NUM_PER_CLASS = min([len(x) for x in mali_classes_images]) -
                    TEST_PER_CLASS - VALID_PER_CLASS

# create_datasets file system
def create_dataset(dataset_folder_name, set_size):
    for cls in range(len(mali_classes_images)):
        f = os.path.join(PROJECT_ROOT, dataset_folder_name, str(cls))
        try:
            rmdir_(f)
        except FileNotFoundError as e:
            pass
        os.makedirs(f)
        img_dir = os.path.join(PROJECT_ROOT, str(cls))
        for img in random.sample(os.listdir(img_dir), set_size):
            img = os.path.join(PROJECT_ROOT, str(cls), img)
            move(img, f)


create_dataset("Training", NUM_PER_CLASS)
create_dataset("Testing", TEST_PER_CLASS)
create_dataset("Validating", VALID_PER_CLASS)

def load_image_tensor(folder):
    """
    folder: image data folder
    """
    image_files = os.listdir(folder)
    image_tensor = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM),
                              dtype=np.float32)
    image_idx = 0
    print(folder)
    for image in image_files:
        image_file = os.path.join(folder, image)
        image_data = (ndimage.imread(image_file).astype(float) -
                      IMAGE_DEPTH / 2) / IMAGE_DEPTH
        image_tensor[image_idx, :, :] = image_data
        image_idx += 1

    return image_tensor

def pickle_tensor(dataset_folder, force=False):
    """
    dataset_name: name of pickled file
    data_dir: image folder
    """
    dataset_names = []
    for level_folder in os.listdir(dataset_folder):
        pickle_name = level_folder + '.pickle'
        pickle_name = os.path.join(dataset_folder, pickle_name)
        dataset_names.append(pickle_name)
        if os.path.isfile(pickle_name) and not force:
            # override by setting force=True.
            print('%s already present - Skipping pickling.' % pickle_name)
        else:
            print('Pickling %s.' % pickle_name)
            image_tensor = load_image_tensor(os.path.join(dataset_folder, level_folder))
            try:
                with open(pickle_name, 'wb') as f:
                    pickle.dump(image_tensor, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', pickle_name, ':', e)
    return dataset_names

TRAIN_SETS_NAME = pickle_tensor(os.path.join(PROJECT_ROOT, "Training"))
TEST_SETS_NAME = pickle_tensor(os.path.join(PROJECT_ROOT, "Testing"))
VALID_SETS_NAME = pickle_tensor(os.path.join(PROJECT_ROOT, "Validating"))

## Shuffle data

def merge_datasets(pickle_files, length):
    """
    label regulation:
    0: benign
    1: mali
    """
    dataset = np.ndarray((length, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    labels = np.ndarray(length, dtype=np.int32)
    # (label, pickle_file)
    current_idx = 0

    for label, pickle_file in enumerate(sorted(pickle_files)):
        try:
            with open(pickle_file, 'rb') as f:
                set1 = pickle.load(f)
            dataset[current_idx: current_idx + set1.shape[0]] = set1
            labels[current_idx: current_idx + set1.shape[0]] = label
            current_idx += set1.shape[0]
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = merge_datasets(TRAIN_SETS_NAME, 5 * NUM_PER_CLASS)
valid_dataset, valid_labels = merge_datasets(VALID_SETS_NAME, 5 * VALID_PER_CLASS)
test_dataset, test_labels = merge_datasets(TEST_SETS_NAME, 5 * TEST_PER_CLASS)

with open("lung_cancer_data.pickle", "wb") as f:
    save = {
        "train_dataset": train_dataset,
        "train_labels": train_labels,
        "valid_dataset": valid_dataset,
        "valid_labels": valid_labels,
        "test_dataset": test_dataset,
        "test_labels": test_labels
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

### END of Data Preparation ###
