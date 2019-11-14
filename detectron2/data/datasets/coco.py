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

        meta.keypoint_connection_rules = [
            # Short sleeve top +0 Lightblue (0,191,255)
            ("1", "2", (0, 191, 255)), ("2", "3", (0, 191, 255)), ("3", "4", (0, 191, 255)), ("4", "5", (0, 191, 255)), ("5", "6", (0, 191, 255)), ("6", "1", (0, 191, 255)),
            ("2", "7", (0, 191, 255)), ("7", "8", (0, 191, 255)), ("8", "9", (0, 191, 255)), ("9", "10", (0, 191, 255)), ("10", "11", (0, 191, 255)),
            ("11", "12", (0, 191, 255)), ("12", "13", (0, 191, 255)), ("13", "14", (0, 191, 255)), ("14", "15", (0, 191, 255)), ("15", "16", (0, 191, 255)),
            ("16", "17", (0, 191, 255)), ("17", "18", (0, 191, 255)), ("18", "19", (0, 191, 255)), ("19", "20", (0, 191, 255)), ("20", "21", (0, 191, 255)), ("21", "22", (0, 191, 255)),
            ("22", "23", (0, 191, 255)), ("23", "24", (0, 191, 255)), ("24", "25", (0, 191, 255)), ("25", "6", (0, 191, 255)),
            # Long sleeve top +25 Green rgb(0,128,0)
            ("26", "27", (0,128,0)), ("27", "28", (0,128,0)), ("28", "29", (0,128,0)), ("29", "30", (0,128,0)), ("30", "31", (0,128,0)), ("31", "26", (0,128,0)),
            ("27", "32", (0,128,0)), ("32", "33", (0,128,0)), ("33", "34", (0,128,0)), ("34", "35", (0,128,0)), ("35", "36", (0,128,0)), ("36", "37", (0,128,0)),
            ("37", "38", (0,128,0)), ("38", "39", (0,128,0)), ("39", "40", (0,128,0)), ("40", "41", (0,128,0)), ("41", "42", (0,128,0)), ("42", "43", (0,128,0)),
            ("43", "44", (0,128,0)), ("44", "45", (0,128,0)), ("45", "46", (0,128,0)), ("46", "47", (0,128,0)), ("47", "48", (0,128,0)), ("48", "49", (0,128,0)),
            ("49", "50", (0,128,0)), ("50", "51", (0,128,0)), ("51", "52", (0,128,0)), ("52", "53", (0,128,0)), ("53", "54", (0,128,0)), ("54", "55", (0,128,0)),
            ("55", "56", (0,128,0)), ("56", "57", (0,128,0)), ("57", "58", (0,128,0)), ("58", "31", (0,128,0)),
            # Short sleeve outwear +58 Yellow rgb(255,255,0)
            ("59", "62", (255,255,0)), ("62", "61", (255,255,0)), ("61", "60", (255,255,0)), ("62", "65", (255,255,0)), ("65", "66", (255,255,0)), ("66", "67", (255,255,0)),
            ("67", "68", (255,255,0)), ("68", "69", (255,255,0)), ("69", "70", (255,255,0)), ("70", "71", (255,255,0)), ("71", "72", (255,255,0)), ("72", "73", (255,255,0)),
            ("73", "74", (255,255,0)), ("74", "89", (255,255,0)), ("89", "88", (255,255,0)), ("88", "60", (255,255,0)), ("64", "59", (255,255,0)), ("64", "63", (255,255,0)),
            ("63", "84", (255,255,0)), ("84", "85", (255,255,0)), ("85", "86", (255,255,0)), ("86", "87", (255,255,0)), ("87", "75", (255,255,0)), ("75", "76", (255,255,0)),
            ("76", "77", (255,255,0)), ("77", "78", (255,255,0)), ("78", "79", (255,255,0)), ("79", "80", (255,255,0)), ("80", "81", (255,255,0)), ("81", "82", (255,255,0)),
            ("82", "83", (255,255,0)), ("83", "64", (255,255,0)),
            # Long sleeve outwear +89 Red rgb(255,0,0)
            ("90", "91", (255,0,0)), ("91", "92", (255,0,0)), ("92", "93", (255,0,0)), ("91", "96", (255,0,0)), ("96", "97", (255,0,0)), ("97", "98", (255,0,0)),
            ("98", "99", (255,0,0)), ("99", "100", (255,0,0)), ("100", "101", (255,0,0)), ("101", "102", (255,0,0)), ("102", "103", (255,0,0)), ("103", "104", (255,0,0)),
            ("104", "105", (255,0,0)), ("105", "106", (255,0,0)), ("106", "107", (255,0,0)), ("107", "108", (255,0,0)), ("108", "109", (255,0,0)), ("109", "128", (255,0,0)),
            ("128", "127", (255,0,0)), ("127", "93", (255,0,0)), ("95", "90", (255,0,0)), ("95", "94", (255,0,0)), ("94", "123", (255,0,0)), ("123", "124", (255,0,0)),
            ("124", "125", (255,0,0)), ("125", "126", (255,0,0)), ("126", "110", (255,0,0)), ("110", "111", (255,0,0)), ("111", "112", (255,0,0)), ("112", "113", (255,0,0)),
            ("113", "114", (255,0,0)), ("114", "115", (255,0,0)), ("115", "116", (255,0,0)), ("116", "117", (255,0,0)), ("117", "118", (255,0,0)), ("118", "119", (255,0,0)),
            ("119", "120", (255,0,0)), ("120", "121", (255,0,0)), ("121", "122", (255,0,0)), ("122", "95", (255,0,0)),
            # Vest +128 DarkOrange rgb(255,140,0)
            ("129", "130", (255,140,0)), ("130", "131", (255,140,0)), ("131", "132", (255,140,0)), ("132", "133", (255,140,0)), ("133", "136", (255,140,0)), ("136", "129", (255,140,0)),
            ("130", "135", (255,140,0)), ("135", "136", (255,140,0)), ("136", "137", (255,140,0)), ("137", "138", (255,140,0)), ("138", "139", (255,140,0)), ("139", "140", (255,140,0)),
            ("140", "141", (255,140,0)), ("141", "142", (255,140,0)), ("142", "143", (255,140,0)), ("143", "134", (255,140,0)),
            # Sling +143 DeepPink rgb(255,20,147)
            ("144", "145", (255,20,147)), ("145", "146", (255,20,147)), ("146", "147", (255,20,147)), ("147", "148", (255,20,147)), ("148", "149", (255,20,147)), ("149", "144", (255,20,147)),
            ("145", "150", (255,20,147)), ("145", "151", (255,20,147)), ("151", "152", (255,20,147)), ("152", "153", (255,20,147)), ("153", "154", (255,20,147)), ("154", "155", (255,20,147)),
            ("155", "156", (255,20,147)), ("156", "157", (255,20,147)), ("157", "149", (255,20,147)), ("149", "158", (255,20,147)),
            # Shorts +158 SaddleBrown rgb(139,69,19)
            ("159", "160", (139,69,19)), ("160", "161", (139,69,19)), ("159", "162", (139,69,19)), ("162", "163", (139,69,19)), ("163", "164", (139,69,19)), ("164", "165", (139,69,19)),
            ("165", "166", (139,69,19)), ("166", "167", (139,69,19)), ("167", "168", (139,69,19)), ("168", "161", (139,69,19)),
            # Trousers +168 Magenta rgb(255,0,255)
            ("169", "170", (255,0,255)), ("170", "171", (255,0,255)), ("169", "172", (255,0,255)), ("172", "173", (255,0,255)), ("173", "174", (255,0,255)), ("174", "175", (255,0,255)),
            ("175", "176", (255,0,255)), ("176", "177", (255,0,255)), ("177", "178", (255,0,255)), ("179", "180", (255,0,255)), ("180", "181", (255,0,255)), ("181", "182", (255,0,255)),
            ("182", "171", (255,0,255)),
            # Skirt +182 GoldenRod rgb(218,165,32)
            ("183", "184", (218,165,32)), ("184", "185", (218,165,32)), ("183", "186", (218,165,32)), ("186", "187", (218,165,32)), ("187", "188", (218,165,32)), ("188", "189", (218,165,32)),
            ("189", "190", (218,165,32)), ("190", "185", (218,165,32)),
            # Short sleeve dress +190 Gray rgb(128,128,128)
            ("191", "192", (128,128,128)), ("192", "193", (128,128,128)), ("193", "194", (128,128,128)), ("194", "195", (128,128,128)), ("195", "196", (128,128,128)), ("196", "191", (128,128,128)),
            ("192", "197", (128,128,128)), ("197", "198", (128,128,128)), ("199", "200", (128,128,128)), ("200", "201", (128,128,128)), ("201", "202", (128,128,128)), ("202", "203", (128,128,128)),
            ("203", "204", (128,128,128)), ("205", "206", (128,128,128)), ("206", "207", (128,128,128)), ("208", "209", (128,128,128)), ("210", "211", (128,128,128)), ("211", "212", (128,128,128)),
            ("212", "213", (128,128,128)), ("213", "214", (128,128,128)), ("214", "215", (128,128,128)), ("216", "217", (128,128,128)), ("217", "218", (128,128,128)), ("218", "219", (128,128,128)),
            ("219", "196", (128,128,128)),
            # Long sleeve dress +219 Darkblue rgb(0,0,139)
            ("220", "221", (0,0,139)), ("221", "222", (0,0,139)), ("222", "223", (0,0,139)), ("223", "224", (0,0,139)), ("224", "225", (0,0,139)), ("225", "220", (0,0,139)),
            ("221", "226", (0,0,139)), ("226", "227", (0,0,139)), ("227", "228", (0,0,139)), ("228", "229", (0,0,139)), ("229", "230", (0,0,139)), ("230", "231", (0,0,139)),
            ("231", "232", (0,0,139)), ("232", "233", (0,0,139)), ("233", "234", (0,0,139)), ("234", "235", (0,0,139)), ("235", "236", (0,0,139)), ("236", "237", (0,0,139)),
            ("237", "238", (0,0,139)), ("238", "239", (0,0,139)), ("239", "240", (0,0,139)), ("240", "241", (0,0,139)), ("241", "242", (0,0,139)), ("242", "243", (0,0,139)),
            ("243", "244", (0,0,139)), ("244", "245", (0,0,139)), ("245", "246", (0,0,139)), ("246", "247", (0,0,139)), ("247", "248", (0,0,139)), ("248", "249", (0,0,139)),
            ("249", "250", (0,0,139)), ("250", "251", (0,0,139)), ("251", "252", (0,0,139)), ("252", "253", (0,0,139)), ("253", "254", (0,0,139)), ("254", "255", (0,0,139)),
            ("255", "256", (0,0,139)),
            # Vest dress +256 Palevioletred rgb(219,112,147)
            ("257", "258", (219,112,147)), ("258", "259", (219,112,147)), ("259", "260", (219,112,147)), ("260", "261", (219,112,147)), ("261", "262", (219,112,147)), ("262", "257", (219,112,147)),
            ("258", "263", (219,112,147)), ("263", "264", (219,112,147)), ("264", "265", (219,112,147)), ("265", "266", (219,112,147)), ("266", "267", (219,112,147)), ("267", "268", (219,112,147)),
            ("268", "269", (219,112,147)), ("269", "270", (219,112,147)), ("270", "271", (219,112,147)), ("271", "272", (219,112,147)), ("272", "273", (219,112,147)), ("273", "274", (219,112,147)),
            # Sling dress +275 Bisque rgb(255, 228, 196)
            ("276", "277", (255, 228, 196)), ("277", "278", (255, 228, 196)), ("278", "279", (255, 228, 196)), ("279", "280", (255, 228, 196)), ("280", "281", (255, 228, 196)), ("281", "276", (255,228, 196)),
            ("277", "282", (255, 228, 196)), ("277", "283", (255, 228, 196)), ("283", "284", (255, 228, 196)), ("284", "285", (255, 228, 196)), ("285", "286", (255, 228, 196)), ("286", "287", (255,228, 196)),
            ("287", "288", (255, 228, 196)), ("288", "289", (255, 228, 196)), ("289", "290", (255, 228, 196)), ("290", "291", (255, 228, 196)), ("291", "292", (255, 228, 196)), ("292", "293", (255,228, 196)),
            ("293", "294", (255, 228, 196))
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
