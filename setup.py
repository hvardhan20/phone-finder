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
        print(e, "Installing MaskRCNN")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+git://github.com/matterport/Mask_RCNN.git"])
    print('Installing requirements')
    try:
        req_file = sys.argv[1]
    except Exception as e:
        print(e, 'Defaulting to "requirements.txt"')
        req_file = 'requirements.txt'
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    print('Setup complete')


install_dependencies()

