import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml
import cv2
import json
import random

def polygons_to_binary_mask(polygons, image_shape):
    if len(image_shape) > 2:
        image_shape = image_shape[:2]

    binary_mask = np.zeros(image_shape, dtype=np.uint8)

    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
        if not np.all((pts >= 0) & (pts < np.array(image_shape[::-1]))):
            continue

        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(binary_mask, [pts], 1)
    
    return binary_mask

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class DatasetSegmentation(Dataset):
    def __init__(self, config_file: dict, processor, mode: str):
        super().__init__()

        self.main_dir = config_file["DATASET"]["MAIN_DIR"]

        if not os.path.isdir(self.main_dir):
            print('-----------------------------')
            print('Main Directory Does Not Exist')
            print('-----------------------------')
            exit()

        self.ending = config_file["DATASET"]["IMAGE_FORMAT"]
        self.mode = mode
        self.files_idx = []

        if self.mode == "train":
            train_txt = config_file["DATASET"]["TRAIN_TXT"]
            with open(train_txt, 'r') as file:
                for line in file:
                    self.files_idx.append(os.path.join(self.main_dir, line.strip()))
        else:
            test_txt = config_file["DATASET"]["TEST_TXT"]
            with open(test_txt, 'r') as file:
                for line in file:
                    self.files_idx.append(os.path.join(self.main_dir, line.strip()))

        self.processor = processor

    def __len__(self):
        return len(self.files_idx)
    
    def get_item(self, index):
        img_idx = self.files_idx[index]
        image_path = f"{img_idx}-color{self.ending}"
        meta_path = f"{img_idx}-meta.json"

        if os.path.exists(image_path) and os.path.exists(meta_path):
            pass
        else:
            return None

        image = Image.open(image_path)
        image_np = np.array(image)
        img_size = image_np.shape
        original_size = tuple(image.size)[::-1]

        pass_list = []
        polygon_list = []
        meta = read_json_file(f"{img_idx}-meta.json")
        for key in meta.keys():
            item = meta[key]
            if not item['found']:
                continue
            pass_list.append(item)
            polygon_list.append(item['segmentation'][0])

        
        random_item = random.choice(pass_list)

        ground_truth_mask = polygons_to_binary_mask(polygon_list, img_size)
        ground_truth_mask_bool = ground_truth_mask.astype(bool)

        x, y, w, h = random_item['bbox_2D_tight']
        bbox = [x, y, x+w, y+h]

        inputs = self.processor(image, original_size, bbox)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask_bool)

        return inputs
    
    def __getitem__(self, index: int) -> dict:
        try:
            data = self.get_item(index)
            if data == None:
                index = random.randint(0, len(self.files_idx) - 1)
                return(self.__getitem__(index))
        except:
            index = random.randint(0, len(self.files_idx) - 1)
            data = self.__getitem__(index)
        return data
    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)