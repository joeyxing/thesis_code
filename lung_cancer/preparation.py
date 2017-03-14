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
        is_mali = False
        if tmp_file_label[1] == 'true':
            is_mali = True
        MALI_LABEL_DICT[tmp_file_label[0]] = is_mali

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
    benign_dir = os.path.join(PROJECT_ROOT, "benign")
    mali_dir = os.path.join(PROJECT_ROOT, "mali")
    if os.path.isdir(benign_dir) and os.path.isdir(mali_dir) and not force:
        print("Dataset already exists, skip creating...")
        return (os.listdir(benign_dir), os.listdir(mali_dir))
    else:
        try:
            print("Try removing existing directory...")
            rmdir_(benign_dir)
            rmdir_(mali_dir)
        except FileNotFoundError as err:
            pass
        
        os.mkdir(benign_dir)
        os.mkdir(mali_dir)
        for image in os.listdir(from_data):
            if MALI_LABEL_DICT[image]:
                copy2(os.path.join(from_data, image), mali_dir)
            else:
                copy2(os.path.join(from_data, image), benign_dir)
        return (os.listdir(benign_dir), os.listdir(mali_dir))

BENIGN_IMAGES, MALI_IMAGES = separate_BM()

TEST_SET_SIZE = 200
VALID_SET_SIZE = 200
TRAIN_SET_SIZE = len(BENIGN_IMAGES) - TEST_SET_SIZE - VALID_SET_SIZE

# create_datasets file system
def create_dataset(dataset_folder_name, set_size):
    bf = os.path.join(PROJECT_ROOT, dataset_folder_name, "benign")
    mf = os.path.join(PROJECT_ROOT, dataset_folder_name, "mali")
    os.makedirs(bf)
    os.makedirs(mf)

    for bi in random.sample(os.listdir(os.path.join(PROJECT_ROOT, "benign")), set_size):
        move(os.path.join(PROJECT_ROOT, "benign", bi), bf)
    for mi in random.sample(os.listdir(os.path.join(PROJECT_ROOT, "mali")), set_size):
        move(os.path.join(PROJECT_ROOT, "mali", mi), mf)

create_dataset("Training", TRAIN_SET_SIZE)
create_dataset("Testing", TEST_SET_SIZE)
create_dataset("Validating", VALID_SET_SIZE)

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
    for bmfolder in os.listdir(dataset_folder):
        pickle_name = bmfolder + '.pickle'
        pickle_name = os.path.join(dataset_folder, pickle_name)
        dataset_names.append(pickle_name)
        if os.path.isfile(os.path.join(dataset_folder, bmfolder, pickle_name)) and not force:
            # override by setting force=True.
            print('%s already present - Skipping pickling.' % pickle_name)
        else:
            print('Pickling %s.' % pickle_name)
            image_tensor = load_image_tensor(os.path.join(dataset_folder, bmfolder))
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

train_dataset, train_labels = merge_datasets(TRAIN_SETS_NAME, 2 * TRAIN_SET_SIZE)
valid_dataset, valid_labels = merge_datasets(VALID_SETS_NAME, 2 * VALID_SET_SIZE)
test_dataset, test_labels = merge_datasets(TEST_SETS_NAME, 2 * TEST_SET_SIZE)

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
