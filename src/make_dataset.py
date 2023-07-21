import argparse
import copy
import json
import os

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make the COCO Single Person Keypoints dataset.')
    parser.add_argument('coco_dir', help='Path to the COCO dataset directory.')
    parser.add_argument(
        'save_dir', help='Path to the directory to save the dataset.')
    parser.add_argument('--subset', default='val2017',
                        help='Subset of the COCO dataset to use. val2017 (default) or train2017.')
    parser.add_argument('--min-area', type=int, default=128*128,
                        help='Minimum area of the person bounding box.')
    parser.add_argument('--min-keypoints', type=int, default=10,
                        help='Minimum number of keypoints on the person.')
    return parser.parse_args()


def crop_image(img_path, save_path, bbox):
    image = Image.open(img_path)
    left, upper, right, lower = bbox
    cropped = image.crop((left, upper, right, lower))
    cropped.save(save_path)


def load_annotations(coco_dir, coco_split):
    """ Load the keypoints annotations from the COCO dataset.

    Args:
        coco_dir (str): Path to the COCO dataset directory.
        coco_split (str): Subset of the COCO dataset to use. val2017 or train2017.

    Returns:
        annotations (dict): Dictionary of the annotations.

    Raises:
        ValueError: If the annotations file does not exist.
    """
    annos_path = os.path.join(coco_dir, 'annotations',
                              f'person_keypoints_{coco_split}.json')
    if not os.path.exists(annos_path):
        raise ValueError(f'Annotations file {annos_path} does not exist.')

    with open(annos_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def filter_people(annotations, min_area=128*128, min_keypoints=10):
    filtered_annotations = {}
    for anno in annotations:
        category_id = anno['category_id']
        if category_id != 1:
            continue

        area = anno['area']
        if area < min_area:
            continue

        num_keypoints = anno['num_keypoints']
        if num_keypoints < min_keypoints:
            continue

        image_id = anno['image_id']
        if image_id not in filtered_annotations:
            keypoints = anno['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]
            bbox = anno['bbox']
            bbox = [float(round(b)) for b in bbox]
            anno['bbox'] = bbox
            for i in range(0, len(keypoints), 3):
                keypoints[i] -= bbox[0]
                keypoints[i+1] -= bbox[1]
            anno['keypoints'] = keypoints
            bbox = anno['bbox']
            anno['segmentation'] = []
            filtered_annotations[image_id] = anno
        else:
            if num_keypoints > filtered_annotations[image_id]['num_keypoints'] or \
                    area > filtered_annotations[image_id]['area']:
                keypoints = anno['keypoints']  # [x1, y1, v1, x2, y2, v2, ...]
                bbox = anno['bbox']
                bbox = [float(round(b)) for b in bbox]
                anno['bbox'] = bbox
                for i in range(0, len(keypoints), 3):
                    keypoints[i] -= bbox[0]
                    keypoints[i+1] -= bbox[1]
                anno['keypoints'] = keypoints
                anno['segmentation'] = []
                filtered_annotations[image_id] = anno

    return filtered_annotations


def main(args):
    # Parse the arguments
    coco_dir = args.coco_dir
    coco_split = args.subset
    save_dir = args.save_dir

    # Create save directories for images and annotations
    save_img_dir = os.path.join(save_dir, coco_split)
    save_annos_dir = os.path.join(save_dir, 'annotations')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_annos_dir, exist_ok=True)

    # Load the annotations and create a copy
    annotations = load_annotations(coco_dir, coco_split)
    annotations_sp = copy.deepcopy(annotations)

    # Keep only the largest person in each image
    filtered_annotations = filter_people(annotations["annotations"],
                                         min_area=args.min_area,
                                         min_keypoints=args.min_keypoints)
    annotations_sp['annotations'] = list(filtered_annotations.values())

    # Keep only the images with people
    image_ids = list(filtered_annotations.keys())
    annotations_sp['images'] = [
        img for img in annotations_sp['images'] if img['id'] in image_ids]

    # Update the image widths and heights to the bbox widths and heights
    for i in range(len(annotations_sp['images'])):
        img = annotations_sp['images'][i]
        image_id = img['id']

        for anno in annotations_sp['annotations']:
            if anno['image_id'] == image_id:
                bbox = anno['bbox']
                img['width'] = round(bbox[2])
                img['height'] = round(bbox[3])
                break

    # Crop the images and save them to a new directory
    img_dir = os.path.join(coco_dir, coco_split)
    for anno in annotations_sp['annotations']:
        image_id = anno['image_id']

        img_path = os.path.join(img_dir, str(image_id).zfill(12) + '.jpg')
        save_img_path = os.path.join(save_img_dir, str(image_id).zfill(12) + '.jpg')

        # Crop the image and save it
        # COCO bbox format is [xmin, ymin, width, height]
        # Convert to [left, upper, right, lower]
        bbox = anno['bbox']
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        crop_image(img_path, save_img_path, bbox)

    # Update bboxes to the new cropped image
    for i in range(len(annotations_sp['annotations'])):
        bbox = annotations_sp['annotations'][i]['bbox']
        annotations_sp['annotations'][i]['bbox'] = [0, 0, bbox[2], bbox[3]]

    # Save the detections
    save_annos_path = os.path.join(save_annos_dir,
                                   f'person_keypoints_{coco_split}.json')
    with open(save_annos_path, 'w') as f:
        json.dump(annotations_sp, f)
