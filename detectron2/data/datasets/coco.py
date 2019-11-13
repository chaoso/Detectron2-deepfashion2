# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import logging
import os

from PIL import Image
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import BoxMode
from .. import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_coco_json", "load_sem_seg"]


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # Add keypoint names from categories, all categories have the same 294 keypoint names, which are string numbers from 1-294
        meta.keypoint_names = [keypoint_name for keypoint_name in coco_api.dataset["categories"][0]["keypoints"]]

        # For keypoints, the same flips don't need to add since that is automatically added later, e.g (1,1)
        # Also don't need to add flipped indices, e.g (1,6) and (6,1), only need to add (1,6),
        # since the create_keypoint_hflip_indices adds the flipped counterparts
        meta.keypoint_flip_map = [
            # Short sleeve top +0
            ("2", "6"), ("3", "5"), ("7", "25"), ("8", "24"), ("9", "23"), ("10", "22"), ("11", "21"), ("12", "20"),
            ("13", "19"), ("14", "18"), ("15", "17"),
            # Long sleeve top +25
            ("27", "31"), ("28", "30"), ("32", "58"), ("33", "57"), ("34", "56"), ("35", "55"), ("36", "53"), ("37", "54"),
            ("38", "52"), ("39", "51"), ("40", "50"), ("41", "49"), ("42", "48"), ("43", "47"), ("44", "46"), ("45", "45"),
            # Short sleeve outwear +58
            ("60", "84"), ("61", "63"), ("62", "64"), ("65", "83"), ("66", "82"), ("67", "81"), ("68", "80"),
            ("69", "79"), ("70", "78"), ("71", "77"), ("72", "76"), ("73", "75"), ("74", "87"), ("89", "86"), ("88", "85"),
            # Long sleeve outwear +89
            ("91", "95"), ("92", "94"), ("93", "123"), ("96", "122"), ("97", "121"), ("98", "120"), ("99", "119"),
            ("100", "118"), ("101", "117"), ("102", "116"), ("103", "115"), ("104", "114"), ("105", "113"), ("106", "112"),
            ("107", "111"), ("108", "110"), ("109", "126"), ("128", "125"), ("127", "124"),
            # Vest +128
            ("130", "134"), ("131", "133"), ("135", "143"), ("136", "142"), ("137", "141"), ("138", "140"),
            # Sling +143
            ("145", "149"), ("146", "148"), ("150", "158"), ("151", "157"), ("152", "156"), ("153", "155"),
            # Shorts +158
            ("159", "161"), ("162", "168"), ("163", "167"), ("164", "166"),
            # Trousers +168
            ("169", "171"), ("172", "182"), ("173", "181"), ("174", "180"), ("175", "179"), ("176", "178"),
            # Skirt +182
            ("183", "185"), ("186", "190"), ("187", "189"),
            # Short sleeve dress +190
            ("192", "196"), ("193", "195"), ("197", "219"), ("198", "218"), ("199", "217"), ("200", "216"), ("201", "215"),
            ("202", "214"), ("203", "213"), ("204", "212"), ("205", "211"), ("206", "210"), ("207", "209"),
            # Long sleeve dress +219
            ("221", "225"), ("222", "224"), ("226", "256"), ("227", "255"), ("228", "254"), ("229", "253"), ("230", "252"),
            ("231", "251"), ("232", "250"), ("233", "249"), ("234", "248"), ("235", "247"), ("236", "246"), ("237", "245"),
            ("238", "244"), ("239", "243"), ("240", "242"),
            # Vest dress +256
            ("258", "262"), ("259", "261"), ("263", "275"), ("264", "274"), ("265", "273"), ("266", "272"), ("267", "271"),
            ("268", "270"),
            # Sling dress +275
            ("277", "281"), ("278", "280"), ("282", "294"), ("283", "293"), ("284", "292"), ("285", "291"), ("286", "290"),
            ("287", "289")
        ]

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(list(coco_api.imgs.keys()))
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warn(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        with PathManager.open(gt_path, "rb") as f:
            img = Image.open(f)
            w, h = img.size
        record["height"] = h
        record["width"] = w
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    import numpy as np
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
