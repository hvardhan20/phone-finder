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
        self.model = MaskRCNN(mode='inference', model_dir=PARAMS["training_model_path"], config=self.pred_config)
        self.model.load_weights(self.model_path, by_name=True)

    def find_phones(self, target_image_path, is_dir=False, show=False):
        images_centroids = {}
        images_to_detect = []
        images_to_detect.extend(
            map(lambda x: target_image_path + x, os.listdir(target_image_path))) if is_dir else images_to_detect.append(
            target_image_path)

        for image_file in images_to_detect:
            if image_file.endswith('.jpg'):
                img = cv2.imread(image_file)
                bbox = self._get_phone_bbox(img)
                bboxes_found = len(bbox)
                if bboxes_found == 1:
                    y1, x1, y2, x2 = bbox[0]
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    normed = normalize(*centroid)
                    logger.info(f'Predicted phone location in {image_file} at {normed}')
                    images_centroids[image_file] = normed
                    if show:
                        show_prediction(img, centroid, normed=normed, images_dir=target_image_path,
                                        image_file=image_file, is_dir=is_dir)
                elif bboxes_found > 1:
                    logger.info(f'Predicted {bboxes_found} phones in {image_file}')
                    images_centroids[image_file] = f"Found {bboxes_found}"
                else:
                    logger.info(f'No phones located in {image_file}')
                    images_centroids[image_file] = 0
        return images_centroids

    def _get_phone_bbox(self, image):
        scaled_image = mold_image(image, self.pred_config)
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        prediction = self.model.detect(sample, verbose=0)[0]
        return prediction['rois']


def show_prediction(img, centroid, normed=None, images_dir=None, image_file=None, is_dir=False):
    if not is_dir:
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        img = cv2.circle(img, centroid, PARAMS["predicted_radius"], (0, 0, 255), 1)
        plt.imshow(img)
        plt.show()
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (350, 320)
        font_scale = 0.6
        font_color = (0, 0, 0)
        line_type = 2
        img = cv2.circle(img, centroid, PARAMS["predicted_radius"], (0, 0, 255), 1)
        cv2.putText(img, ' '.join([str(i) for i in normed]),
                    bottom_left_corner,
                    font,
                    font_scale,
                    font_color,
                    line_type)
        cv2.imwrite(images_dir + '_' + os.path.basename(image_file), img)


def main(args):
    try:
        path = args.path
    except Exception as e:
        logger.error('Path to image not provided', e)
        exit(1)
    if os.path.isdir(path):
        path += '/'
    elif os.path.isfile(path) and path[-1] == '/':
        path = path[:-1]
    model_file = PARAMS['trained_weights_file_path']
    if not model_file:
        logger.error("Enter a valid path for trained model file")
        exit(1)
    logger.info(f'Loading model from file {model_file}')
    finder = PhoneFinder(model_file)
    phone_centroids = finder.find_phones(path, is_dir=os.path.isdir(path), show=PARAMS["show_predictions"])
    if len(phone_centroids) == 1:
        center = list(phone_centroids.values())[0]
        if center:
            print(center[0], center[1])
        else:
            logger.info('Phone not found')
    elif not phone_centroids:
        print('No phones found')
    else:
        import json
        with open('inference.json', 'wt+') as f:
            json.dump(phone_centroids, f, indent=4)
        print(json.dumps(phone_centroids, indent=4))


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
