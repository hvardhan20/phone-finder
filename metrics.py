"""
Written by Harshavardhan
"""
import logging
import sys
from global_config import PARAMS
from phone_config_dataset import PhoneConfig, PhoneDataSet, PredictionConfig
from numpy import expand_dims
from numpy import mean
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from mrcnn.utils import compute_ap, compute_recall
from mrcnn import visualize

logger = logging.getLogger(__name__)
logger.disabled = PARAMS['disable_trainer_logging']

APs = []
recalls = []


def evaluate_model(dataset, model, cfg):
    global APs, recalls
    APs = []
    recalls = []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        recall, ids = compute_recall(r["rois"], gt_bbox, 0.75)
        APs.append(AP)
        recalls.append(recall)
    mAP = mean(APs)
    return mAP, mean(recalls)

try:
    path = sys.argv[1]
except:
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
train_mean_abs_precision, recalls = evaluate_model(trainset, model, config)
print("Train mean absolute precision: %.3f recalls: %.3f" % (train_mean_abs_precision, recalls))
# evaluate model on test dataset
test_mean_abs_precision, test_recalls = evaluate_model(testset, model, config)
print("Test mean absolute precision: %.3f recalls: %.3f" % (test_mean_abs_precision, test_recalls))
print(APs)
print(recalls)
import matplotlib.pyplot as plt
_, ax = plt.subplots(1)
ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(0.75))
ax.set_ylim(0, 1.1)
ax.set_xlim(0, 1.1)
_ = ax.plot(recalls, APs)
plt.show()
plt.savefig('recall-precision.png')