import shutil
import time
import os

tic = time.time()

NUMBER_OF_SAMPLES_TRAIN = 1000
NUMBER_OF_SAMPLES_VALIDATION = 300

TRAINING_IMAGE_FILE_PATH = "F:\\Downloads\\train\\train\\image"
VALIDATION_IMAGE_FILE_PATH = "F:\\Downloads\\validation\\validation\\image"

DST_VALIDATION_IMAGE_FILE_PATH = "F:\\mini_deepfashion2\\validation\\image"
DST_TRAINING_IMAGE_FILE_PATH = "F:\\mini_deepfashion2\\train\\image"

i = 0
for path, _, files in os.walk(TRAINING_IMAGE_FILE_PATH):
    for file in files:
        if i == NUMBER_OF_SAMPLES_TRAIN:
            break
        shutil.copy(os.path.join(path, file), DST_TRAINING_IMAGE_FILE_PATH)
        i += 1
    break

j = 0
for path, _, files in os.walk(VALIDATION_IMAGE_FILE_PATH):
    for file in files:
        if j == NUMBER_OF_SAMPLES_VALIDATION:
            break
        shutil.copy(os.path.join(path, file), DST_VALIDATION_IMAGE_FILE_PATH)
        j += 1
    break


print('Done (t={:0.2f}s)'.format(time.time() - tic))
