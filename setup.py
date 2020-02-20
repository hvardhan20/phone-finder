"""
Author: Harshavardhan
"""
import subprocess
import sys
import os


def install_dependencies():
    try:
        from mrcnn.model import MaskRCNN
    except Exception as e:
        print(e)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/matterport/Mask_RCNN.git"])
    print('Installing requirements')
    try:
        req_file = sys.argv[1]
    except:
        req_file = 'requirements.txt'
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    print('Setup complete')


install_dependencies()

