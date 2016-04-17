"""This file will detect all the images in a given directory and if a
face is detected in an image it will cut the biggest face detected. This
face image is then in a separate folder wit the same image name.
The saved image it is not only the square detected around the face but
it tries to maximize the space around the face and and at the same time
keeping a square ratio.
"""

#!/bin/env python
#build_dataset.py

import os
from face_detection.dataset_builder import DatasetBuilder

# Path to read images from
ROOT_PATH = ''
# path to put the resulting images in
DST_PATH = ''

if not os.path.isdir(DST_PATH):
    os.mkdir(DST_PATH)

# specify which folders inside ROOT_PATH to look into
FOLDERS = ['old']
EXTENSIONS = ['png', 'jpg']
# maximum amount of images to look for
MAX_IMAGES = 10
MAX_NUMBER_OF_FACES_PER_IMAGE = 2

FACES_BUILDER = DatasetBuilder(ROOT_PATH,
                               DST_PATH,
                               MAX_NUMBER_OF_FACES_PER_IMAGE)
IMAGES_PATH = FACES_BUILDER.get_images_path(FOLDERS,
                                            EXTENSIONS,
                                            MAX_IMAGES)
FACES, LABELS = FACES_BUILDER.get_faces(IMAGES_PATH)
FACES_BUILDER.save_images(FACES, LABELS)
