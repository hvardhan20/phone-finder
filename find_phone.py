"""
Author: Harshavardhan
"""
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import os
from global_config import PARAMS, update_params_with_latest_model
from phone_config_dataset import PredictionConfig, normalize
from mrcnn.model import MaskRCNN, mold_image

logger = logging.getLogger(__name__)
logger.disabled = PARAMS['disable_find_phone_logging']


class PhoneFinder:
    def __init__(self, model_file):
        self.model_path = model_file
        self.pred_config = PredictionConfig()
        self.model = MaskRCNN(mode='inference', model_dir='./', config=self.pred_config)
        self.model.load_weights(self.model_path, by_name=True)

    def find_phone(self, image_path, show=False):
        img = cv2.imread(image_path)
        bbox = self._get_phone_bbox(img)
        if len(bbox) == 1:
            bbox = bbox[0]
            y1, x1, y2, x2 = bbox
            if show:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                plt.imshow(img)
                plt.show()
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            return centroid

    def find_phones(self, images_dir, show=False):
        images_centroids = {}
        for image_file in os.listdir(images_dir):
            if image_file.endswith('.jpg'):
                img = cv2.imread(images_dir + image_file)
                logger.info(f'Predicting phone location in {image_file}')
                bbox = self._get_phone_bbox(img)
                bboxes_found = len(bbox)
                if bboxes_found == 1:
                    bbox = bbox[0]
                    y1, x1, y2, x2 = bbox
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    normed = normalize(*centroid)
                    images_centroids[image_file] = normed
                    if show:
                        show_prediction(img, normed, images_dir, image_file, centroid=centroid)
                elif bboxes_found > 1:
                    images_centroids[image_file] = f"Found {bboxes_found}"
                else:
                    images_centroids[image_file] = 0

        return images_centroids

    def _get_phone_bbox(self, image):
        scaled_image = mold_image(image, self.pred_config)
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        prediction = self.model.detect(sample, verbose=0)[0]
        return prediction['rois']


def show_prediction(img, normed, images_dir, image_file, centroid=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (350, 320)
    font_scale = 0.6
    font_color = (0, 0, 0)
    line_type = 2
    img = cv2.circle(img, centroid, 5, (0, 0, 255), 1)
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, ' '.join([str(i) for i in normed]),
                bottom_left_corner,
                font,
                font_scale,
                font_color,
                line_type)
    cv2.imwrite(images_dir + '_' + image_file, img)


def main(args):
    try:
        path = args.path
    except Exception as e:
        logger.error('Path to image not provided', e)
        exit(1)
    if path[-1] != '/' or path[-1] != '\\':
        path += '/'
    model_file = PARAMS['trained_weights_file_path']
    if not model_file:
        logger.error("Enter a valid path for trained model file")
        exit(1)
    logger.info(f'Loading model from file {model_file}')
    finder = PhoneFinder(model_file)
    if os.path.isfile(path):
        logger.info(f'Finding phone in the image {path}')
        phone_centroid = finder.find_phone(path, show=True)
        if phone_centroid:
            norm_centroid = normalize(*phone_centroid)
            print(norm_centroid[0], norm_centroid[1])
        else:
            logger.info('Phone not found')
    else:
        logger.info(f'Finding phones in the images directory {path}')
        res = finder.find_phones(path, show=True)
        import json
        with open('inference.json', 'wt+') as f:
            json.dump(res, f, indent=4)
        print(json.dumps(res, indent=4))


if __name__ == '__main__':
    try:
        # update_params_with_latest_model()
        parser = argparse.ArgumentParser(description="Phone finder")
        parser.add_argument("path", help="Path to the image to be tested")
        args = parser.parse_args()
    except:
        logger.error("Error parsing command line arguments. Please provide an image path")
        exit(1)
    main(args)
