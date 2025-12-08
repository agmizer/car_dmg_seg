from pycocotools.coco import COCO
import numpy as np
import cv2
import os
from PIL import Image

ann_file = '../dataset/train/_annotations.coco.json'
train_coco = COCO(ann_file)
ann_file = '../dataset/valid/_annotations.coco.json'
valid_coco = COCO(ann_file)
ann_file = '../dataset/test/_annotations.coco.json'
test_coco = COCO(ann_file)

def load_data(coco, split):
    image_dir = f'../dataset/{split}'
    image_ids = coco.getImgIds()
    data = []
    for id in image_ids:
        img_info = coco.loadImgs(id)[0]
        width = img_info['width']
        height = img_info['height']
        filename = img_info['file_name']
        image_path = os.path.join(image_dir, filename)
        image = np.array(Image.open(image_path))
        ann_ids = coco.getAnnIds(imgIds=id)
        anns = coco.loadAnns(ann_ids)
        
        mask = np.zeros((height, width), dtype=np.uint8)

        bboxes = []
        categories = []

        for ann in anns:
            category_id = ann['category_id']
            segmentation = ann['segmentation']
            bbox = ann['bbox']

            bboxes.append(bbox)
            categories.append(category_id)

            for polygon in segmentation:
                points = np.array(polygon).reshape(-1, 2)
                cv2.fillPoly(mask, [points], category_id)

        data.append({
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'categories': categories,
            'filename': filename,
            'image_id': id,
        })

    return data

def load_masks():
    train_data = load_data(train_coco, 'train')
    valid_data = load_data(valid_coco, 'valid')
    test_data = load_data(test_coco, 'test')
    return train_data, valid_data, test_data

def get_categories(coco):
    categories = coco.loadCats(coco.getCatIds())
    return {cat['id']: cat['name'] for cat in categories}
