import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv

# Root directory of the project

#ROOT_DIR = os.path.abspath("../")
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append('D:\maskrcnn\Mask_RCNN')  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(ROOT_DIR)
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class ShapesConfig(coco.Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    #IMAGE_MAX_DIM = 1024
    IMAGE_MAX_DIM = 2048

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

class InferenceConfig(coco.CocoConfig):
#class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#实例化一个对象，用于配置展示
#类继承关系，Config->CocoConfig->Inferenceconfig
config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
file_image = "D:\maskrcnn\Mask_RCNN\images\8512296263_5fc5458e20_z.jpg"
file_image2 = r"D:\maskrcnn\Mask_RCNN\1.jpg"
file_image3 = r"D:\maskrcnn\Mask_RCNN\000000.png"
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

image = skimage.io.imread(file_image)
#image3 = skimage.io.imread(file_image3)
image2 = cv.imread(file_image2,cv.IMREAD_COLOR)
image3 = cv.imread(file_image3,cv.IMREAD_COLOR)
#cv.imshow("原图",image2)
#cv.waitKey(0)
# Run detection
results = model.detect([image3], verbose=1)

# Visualize results
r = results[0]

visualize.display_instances(image3, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
print(r['masks'].shape)

image_output=image3

def trans_image(image):
    height,width = image.shape[:2]
    channels = image.shape[2]
    for c in range(channels):
        for row in range(height):
            for col in range(width):
                level = image[:,:]
                image[:,:] = -1
    return image

height,width = image3.shape[:2]

image_output=np.zeros((height,width))
image_output[:,:]=-1

N_class=len(r['class_ids'])
for i in range(N_class):
    mask=r['masks'][:, :, i]
    class_id=r['class_ids'][N_class-1]
    image_output[:,:]=np.where(mask==1,image_output[:,:]+1+class_id,image_output[:,:])
#print(image_output.shape)
#image_output=trans_image(image_output)
print(image_output)