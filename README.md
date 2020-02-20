# Phone Finder

This project provides a model trainer to predict the location of a phone in a given image. 

## Getting Started

The main runnable files in this project are `train_phone_finder.py` and `find_phone.py`

### Prerequisites

Please try to run this project from the root of the project directory as much as possible.
Command line arguments **CAN BE ABSOLUTE**.
From the root of this project directory structure, run the `setup.py` script as 
```
python setup.py
```
to install all the project dependencies. This script installs Mask R-CNN package, 
which is the backbone of this entire project. Mask R-CNN package is installed directly from Git source
repository as its PyPI version is not updated with the latest changes. Hence it is buggy. 
Besides Mask R-CNN, there are other packages that are required and are installed from the `requirements.txt`

### Initializing

This program is highly configurable via the `./config/params.json` file. You can configure the locations of
training checkpoint models, hyperparameters like number of epochs and number of steps 
in each epoch, learning rate, layers to trains and exclude. You can also disable logging in trainer
and finder module. Details about each param is listed below


```
{
    "training_model_path": "<checkpoint path for the model being trained>",
    "trained_weights_file_path": "<Pre-trained COCO weights file for training on top of that and for Prediction>",
    "class_config_name": "<configuration name for the class being identified>",
    "number_of_epoch_steps": <number of steps in each epoch>,
    "number_of_epochs": <number of epochs>,
    "learning_rate": <learning rate to train with. Default is 0.001>,
    "layers": "<layers to train. Default is 'all'>",
    "layers_to_exclude_while_training": ["<List of strings specifying the layers in the pre-trained weights to exlude from training>"],
    "number_of_classes": <Number of classes to detect. Here, its phone and the background (BG is always present)>,
    "bbox_x_offset": <Horizontal offset used to create bounding box around the phone using the annotated center of phone>,,
    "bbox_y_offset": <Vertical offset used to create bounding box around the phone using the annotated center of phone>,
    "test_size": <Test split from the dataset. Train split is calculated accordingly>,
    "random_seed": <Random seed for shuffling dataset>,
    "disable_find_phone_logging": <Boolean value as false or true. true for disable logging>,
    "disable_trainer_logging": <Boolean value as false or true. true for disable logging>,
    "disable_error_reporting": <Boolean value as false or true. true for disable logging>
}
```

## Training the model

Tune the hyperparameters from `./config/params.json` and run `train_phone_finder.py` with path to 
image data set directory with the `labels.txt` inside the path as

```
python train_phone_finder.py ./images
```

##### **CAVEAT**
Model training can take a long time if you're running without a significantly powerful GPU on the system.
Please train the model on a system with CUDA enabled card with compute compatibility higher than 3.5

If you cannot train a model, you can use the model bundled with this program.

## Running the tests - Predictions

After the model is trained, the new model file path is automatically updated in the `./config/params.json`.
Please execute the `find_phone.py` with the path to image or directory of images to be predicted like 
```
python find_phone.py ./images/21.jpg
``` 
Detection of phone does not take as long as training does and is quite fast

