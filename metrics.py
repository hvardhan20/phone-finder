"""
Written by Harshavardhan
"""
import logging
from numpy import zeros
from numpy import asarray
from global_config import PARAMS
from phone_config_dataset import PhoneConfig, PhoneDataSet, PredictionConfig
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset, compute_ap
from mrcnn.model import load_image_gt, mold_image

logger = logging.getLogger(__name__)
logger.disabled = PARAMS['disable_trainer_logging']


def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


path = './images/'
# load the train dataset
trainset = PhoneDataSet(path)
trainset.load_dataset(purpose='train')
trainset.prepare()

testset = PhoneDataSet(path)
testset.load_dataset(purpose='test')
testset.prepare()

config = PredictionConfig()
logger.info('Creating a Mask R-CNN model')
model = MaskRCNN(mode='inference', model_dir=PARAMS['training_model_path'], config=config)
model.keras_model.metrics_tensors = []
logger.info('Loading weights')
if PARAMS['trained_weights_file_path']:
    model_weights = PARAMS['trained_weights_file_path']
else:
    model_weights = model.get_imagenet_weights()
model.load_weights(model_weights, by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(trainset, model, config)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(testset, model, config)
print("Test mAP: %.3f" % test_mAP)
