import os
import sys
import json
import yaml
import logging
import argparse
from coloredlogs import ColoredFormatter

def convert_bbox(size, bbox):
    """
    Convert COCO bbox format (x_min, y_min, width, height) to YOLO format
    (x_center, y_center, width, height), normalized by the dimensions of the image.

    :param size:
    :param bbox:
    :return:
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = bbox[0] + bbox[2] / 2.0
    y_center = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    return x_center * dw, y_center * dh, w * dw, h * dh


def convert_segmentation(size, segmentation):
    """
    Convert COCO segmentation polygons to YOLO format.
    Normalizes the segmentation polygon points by image width and height.

    :param size:
    :param segmentation:
    """
    width, height = size
    normalized_points = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / width
        y = segmentation[i + 1] / height
        normalized_points.append(x)
        normalized_points.append(y)
    return normalized_points


def convert_coco_to_yolo_segmentation(coco_json_path, output_dir):
    """

    :param coco_json_path:
    :param output_dir:
    :return:
    """
    print("Opening COCO file...")
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    category_id_to_class_id = {}
    for i, category in enumerate(data['categories']):
        category_id_to_class_id[category['id']] = i
    print("Categories: {}".format(category_id_to_class_id))

    os.makedirs(output_dir, exist_ok=True)

    print("Building annotations files...")
    for img in data['images']:
        image_id = img['id']
        img_width = img['width']
        img_height = img['height']
        img_filename = img['file_name']

        txt_file_path = os.path.join(output_dir, f"{os.path.splitext(img_filename)[0]}.txt")

        with open(txt_file_path, 'w') as txt_file:
            for ann in data['annotations']:
                if ann['image_id'] == image_id:
                    if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                        normalized_seg = convert_segmentation((img_width, img_height), ann['segmentation'][0])
                        class_id = category_id_to_class_id[ann['category_id']]
                        txt_file.write(f"{class_id} " + " ".join(map(str, normalized_seg)) + "\n")

    print(f"Segmentation YOLO annotations saved in {output_dir}")


def convert_coco_to_yolo(coco_json_path, image_dir, output_dir):
    """

    :param coco_json_path:
    :param image_dir:
    :param output_dir:
    :return:
    """
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    class_names = [categories[i] for i in sorted(categories.keys())]

    print(categories, class_names)

    os.makedirs(output_dir, exist_ok=True)

    for img in data['images']:
        image_id = img['id']
        img_width = img['width']
        img_height = img['height']
        img_filename = img['file_name']

        txt_file_path = os.path.join(output_dir, f"{os.path.splitext(img_filename)[0]}.txt")

        with open(txt_file_path, 'w') as txt_file:
            for ann in data['annotations']:
                if ann['image_id'] == image_id:
                    bbox = convert_bbox((img_width, img_height), ann['bbox'])
                    class_id = ann['category_id'] - 1

                    txt_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    yolo_yaml_data = {
        'train': image_dir,
        'val': image_dir,
        'nc': len(class_names),
        'names': class_names,
    }

    filename = os.path.basename(coco_json_path).split(".")[0]
    yaml_file_path = os.path.join(output_dir, filename + ".yaml")
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yolo_yaml_data, yaml_file, default_flow_style=False)

    print(f"YOLO annotations and YAML file saved in {output_dir}")

def convert_coco_to_yaml(coco_json_path, image_dir, output_yaml_path):
    """

    :param coco_json_path:
    :param image_dir:
    :param output_yaml_path:
    :return:
    """
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    class_names = [categories[i] for i in sorted(categories.keys())]

    yolo_yaml_data = {
        'train': image_dir,
        'val': image_dir,
        'nc': len(class_names),
        'names': class_names,
    }

    with open(output_yaml_path, 'w') as outfile:
        yaml.dump(yolo_yaml_data, outfile, default_flow_style=False)

    print(f"YAML file saved at {output_yaml_path}")


def main(arguments):
    coco_json_path = arguments.coco_json_path
    image_dir = arguments.image_dir
    output_filepath = arguments.output_filepath

    output_dir = os.path.dirname(output_filepath)

    convert_coco_to_yolo_segmentation(coco_json_path, output_dir)

if __name__ == '__main__':
    """
    Example:
        > python main.py -coco_json_path STRING_PATH -image_dir STRING_PATH -output_filepath STRING_PATH -verbose BOOLEAN

    Usage:
        > python main.py -coco_json_path /media/rodolfo/data/personal/livecell/data/livecell/annotations/LIVECell/livecell_coco_test.json
                         -image_dir /content/drive/My Drive/livecells/images/livecell_test_images/
                         -output_filepath /media/rodolfo/data/personal/livecell/data/livecell/annotations/LIVECell/yolo/livecell_coco_test.yaml
                         -verbose True
    """
    parser = argparse.ArgumentParser(description='ThreeV Assessment')
    parser.add_argument('-coco_json_path', action="store", dest='coco_json_path', help='REF')
    parser.add_argument('-image_dir', action="store", dest='image_dir', help='REF')
    parser.add_argument('-output_filepath', action="store", dest='output_filepath', help='REF')
    parser.add_argument('-verbose', action="store", dest='verbose', help='Print log of processing')
    args = parser.parse_args()

    if eval(args.verbose):
        log = logging.getLogger('')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cf = ColoredFormatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ")
        ch.setFormatter(cf)
        log.addHandler(ch)

        fh = logging.FileHandler('logging.log')
        fh.setLevel(logging.INFO)
        ff = logging.Formatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ",
                               datefmt='%Y.%m.%d %H:%M:%S')
        fh.setFormatter(ff)
        log.addHandler(fh)

        log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")

    main(args)