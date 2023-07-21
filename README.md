# COCO Single Person Keypoints Dataset

Welcome to the repository of the COCO Single Person Keypoints Dataset. This is a processed subset of the famous COCO Dataset, where each image contains exactly one person. The dataset is particularly suitable for tasks related to human pose estimation.

## Dataset Creation
The dataset is created by processing the original COCO 2017 dataset. It extracts only those images where there's exactly one person in the image. The images are cropped to the bounding box of the person, and the annotations are updated accordingly.

## Structure
The dataset follows the same structure as the original COCO dataset. There are three main types of data:

- Images: Located in the `data/coco_single_person/val2017` directory.
- Annotations: The annotations file is found at `data/coco_single_person/annotations/person_keypoints_val2017.json`.
- Image info: Additional information about the images can be found in the annotations file.

## Usage
The dataset can be used in a similar way as the COCO dataset. For Python users, we recommend the [pycocotools](https://github.com/cocodataset/cocoapi) library which provides API to read and manipulate the data. Check out [this notebook](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb) for a quick introduction to using pycocotools with a COCO styled dataset.

## Evaluation
You can evaluate your pose estimation models on this dataset using the COCO Evaluation metric. An example code for this is given in this repository.

## Contributions
This repository is open to contributions. If you encounter any issues while using the dataset or have suggestions for improvements, please open an issue. For major changes, please open an issue first to discuss what you would like to change.