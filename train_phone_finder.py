"""
Author: Harshavardhan
"""
import argparse
import logging
from global_config import PARAMS, update_params_with_latest_model
from phone_config_dataset import PhoneConfig, PhoneDataSet
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from keras.utils.vis_utils import plot_model

logger = logging.getLogger(__name__)
logger.disabled = PARAMS['disable_trainer_logging']


def train_model_from_dataset(path):
    try:
        trainset = PhoneDataSet(path)
        trainset.load_dataset(purpose='train')
        trainset.prepare()

        testset = PhoneDataSet(path)
        testset.load_dataset(purpose='test')
        testset.prepare()

        # Display the masks created over the test and train datasets
        show_dataset_masks(testset)

        config = PhoneConfig()
        logger.info('Creating a Mask R-CNN model')
        model = MaskRCNN(mode='training', model_dir=PARAMS['training_model_path'], config=config)
        model.keras_model.metrics_tensors = []
        logger.info('Loading weights')
        if PARAMS['trained_weights_file_path']:
            model_weights = PARAMS['trained_weights_file_path']
        else:
            model_weights = model.get_imagenet_weights()

        # model.load_weights(model_weights, by_name=True,
        #                    exclude=PARAMS["layers_to_exclude_while_training"])
        logger.info('Starting model training...')
        # model.train(trainset, testset, learning_rate=config.LEARNING_RATE, epochs=PARAMS['number_of_epochs'],
        #             layers=PARAMS["layers"])

        # plot_model(model.keras_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return True
    except Exception as e:
        logger.error("Error while training the model", e)
        return False


def show_dataset_masks(dataset):
    for image_id in range(130):
        try:
            image = dataset.load_image(image_id)
            # load the masks and the class ids
            mask, class_ids = dataset.load_mask(image_id)
            # extract bounding boxes from the masks
            bbox = extract_bboxes(mask)
            # display image with masks and bounding boxes
            display_instances(image, bbox, mask, class_ids, dataset.class_names)
        except:
            continue


def main(args):
    path = args.path
    if not path:
        logging.warning('No path found. Defaulting to ./images')
        path = './images/'
    if path[-1] != '/' or path[-1] != '\\':
        path += '/'

    if train_model_from_dataset(path):
        update_params_with_latest_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phone finder model trainer")
    parser.add_argument("path", help="Data set path to train")
    args = parser.parse_args()
    main(args)
