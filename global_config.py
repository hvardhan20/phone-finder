"""
Author: Harshavardhan
"""
print("At the begining")
import logging.config
from yaml import safe_load
import json
import glob
import os
import sys
import logging

print("here")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
print("before starting")

__PARAMS_PATH = "/content/phone-finder/config/params.json"
__LOGGING_PATH = "/content/phone-finder/config/logging.yaml"

with open(__PARAMS_PATH, "rt") as f:
    print("opened")
    PARAMS = json.load(f)

with open(__LOGGING_PATH, 'rt') as f:
    print("logging")
    config = safe_load(f.read())
    logging.config.dictConfig(config)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').disabled = True
    print("finished logging")


def update_params_with_latest_model():
    global PARAMS
    try:
        # get all dirs starting with the name `class_config_name`
        list_of_dirs = glob.glob(PARAMS['training_model_path']+'/'+PARAMS['class_config_name']+'*/')
        latest_dir = max(list_of_dirs, key=os.path.getctime)
        list_of_models = glob.glob(latest_dir+'/*.h5')  # get all model files
        latest_model = max(list_of_models, key=os.path.getctime)

        with open(__PARAMS_PATH, "rt") as jsonFile:
            data = json.load(jsonFile)
            data["model_file_path"] = latest_model
            PARAMS = data

        with open(__PARAMS_PATH, "wt") as jsonFile:
            json.dump(data, jsonFile, indent=4)
    except Exception as e:
        print("Error updating params with latest model", e)