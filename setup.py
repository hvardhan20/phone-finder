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
    req_file = os.getcwd() + '/requirements.txt'
    print('Installing requirements')
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", sys.argv[1]])
    print('Setup complete')


install_dependencies()

