"""
Author: Harshavardhan
"""
import json
import re
import pandas as pd
import numpy as np
import cv2
import os
from global_config import PARAMS
from mrcnn.utils import Dataset
from mrcnn.config import Config


def normalize(x, y, w=490, h=326):
    return round(x / w, 4), round(y / h, 4)


def denormalize(norm_x, norm_y, w=490, h=326):
    return norm_x * w, norm_y * h


def get_bbox(x, y, x_offset=PARAMS['bbox_x_offset'], y_offset=PARAMS['bbox_y_offset']):
    bbox = [round(x - x_offset) if (x - x_offset) > 0 else 0,
            round(y - y_offset) if (y - y_offset) > 0 else 0,
            round(x + x_offset) if (x + x_offset) > 0 else 0,
            round(y + y_offset) if (y + y_offset) > 0 else 0]
    return [int(i) for i in bbox]


class PredictionConfig(Config):
    NAME = PARAMS['class_config_name']
    NUM_CLASSES = PARAMS["number_of_classes"]
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class PhoneConfig(Config):
    # Distinct configuration name
    NAME = PARAMS['class_config_name']
    # Number of classes. In this case background & phone
    NUM_CLASSES = PARAMS["number_of_classes"]
    # Number of training steps per epoch
    STEPS_PER_EPOCH = PARAMS["number_of_epoch_steps"]
    # Learning rate
    LEARNING_RATE = PARAMS["learning_rate"]


class PhoneDataSet(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        labels_file = path + 'labels.txt'
        labels = {}
        with open(labels_file, 'r') as lf:
            lines = lf.readlines()
            for line in lines:
                line_content = [x for x in re.split('\\s+', line.strip())]
                labels[line_content[0]] = {'x': float(line_content[1]), 'y': float(line_content[2])}
        self.data = pd.read_json(json.dumps(labels), orient='index')
        self.data.index.name = 'file'

    def load_dataset(self, from_path=None, purpose='train', test_size=PARAMS['test_size'], class_name="phone",
                     seed=PARAMS['random_seed']):
        self.add_class(source="dataset", class_id=1, class_name=class_name)
        train_size = 1 - test_size
        images_dir = from_path if from_path else self.path
        dir_npa = np.asarray(os.listdir(images_dir))
        data_set_size = len(dir_npa)
        np.random.seed(seed)
        np.random.shuffle(dir_npa)
        if purpose == 'train':
            n_train = round(train_size * data_set_size)
            fin_data = dir_npa[:n_train]
        else:
            n_test = round(test_size * data_set_size)
            fin_data = dir_npa[-n_test:]
        for filename in fin_data:
            if filename.endswith('.jpg'):
                image_id = filename.split('.')[0]
                img_path = images_dir + filename
                self.add_image('dataset', image_id=image_id, path=img_path, filename=filename)

    def extract_boxes(self, filename):
        x = self.data.loc[filename].x
        y = self.data.loc[filename].y
        img = cv2.imread(self.path + filename)
        img_height, img_width = img.shape[:2]
        return get_bbox(*denormalize(x, y, w=img_width, h=img_height)), img_width, img_height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        box, w, h = self.extract_boxes(info['filename'])
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, 1], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(1):
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('phone'))
        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
