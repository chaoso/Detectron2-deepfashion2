import os
import sys

from detectron2.config import get_cfg


class ClothConfig:
    ROOT_DIR = os.path.abspath("../")
    sys.path.append(ROOT_DIR)

    train_coco_json = "F:\\downloads\\train\\train\\annos_coco\\training_coco.json"
    NAME_OF_DATASET_TRAIN = "deepfashion2_train"
    IMAGE_DIR_TRAIN = "F:\\downloads\\train\\train\image"

    val_coco_json = "F:\\Downloads\\validation\\validation\\annos_coco\\validation_coco.json"
    NAME_OF_DATASET_VAL = "deepfashion2_val"
    IMAGE_DIR_VAL = "F:\\Downloads\\validation\\validation\\image"

    NUMBER_OF_SAMPLES_TRAIN = 100
    NUMBER_OF_SAMPLES_VAL = 30

    cfg = get_cfg()
    cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = (NAME_OF_DATASET_TRAIN,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13  # 13 clothing categories
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
