"""
Author: Harshavardhan
"""
import argparse
import logging
from global_config import PARAMS, update_params_with_latest_model
from phone_config_dataset import PhoneConfig, PhoneDataSet
from mrcnn.model import MaskRCNN

logger = logging.getLogger(__name__)
logger.disabled = PARAMS['trainer_logging_disable']


def train_model_from_dataset(path):
    try:
        trainset = PhoneDataSet(path)
        trainset.load_dataset(purpose='train')
        trainset.prepare()

        testset = PhoneDataSet(path)
        testset.load_dataset(purpose='test')
        testset.prepare()

        config = PhoneConfig()
        logger.info('Creating a Mask R-CNN model')
        model = MaskRCNN(mode='training', model_dir=PARAMS['training_model_path'], config=config)
        model.keras_model.metrics_tensors = []
        logger.info('Loading weights')
        if PARAMS['model_file_path']:
            model_weights = PARAMS['model_file_path']
        else:
            model_weights = model.get_imagenet_weights()
        model.load_weights(model_weights, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        logger.info('Starting model training...')
        model.train(trainset, testset, learning_rate=config.LEARNING_RATE, epochs=PARAMS['number_of_epochs'],
                    layers='heads')
        return True
    except Exception as e:
        logger.error("Error while training the model", e)
        return False


def main(args):
    path = args.path
    if not path:
        logging.warning('No path found. Defaulting to ./images')
        path = './images/'
    if path[-1] != '/' or path[-1] != '\\':
        path += '/'

    # if train_model_from_dataset(path):
    update_params_with_latest_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phone finder model trainer")
    parser.add_argument("path", help="Data set path to train")
    args = parser.parse_args()
    main(args)
