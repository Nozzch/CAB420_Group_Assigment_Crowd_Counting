import numpy as np
import pandas as pd
import os
from glob import glob
import cv2
import json
import pickle


def generate_data(label_path):
    rgb_path = label_path.replace('GT', 'RGB').replace('json', 'jpg')
    t_path = label_path.replace('GT', 'T').replace('json', 'jpg')
    rgb = cv2.imread(rgb_path)[..., ::-1].copy()
    t = cv2.imread(t_path)[..., ::-1].copy()
    im_h, im_w, _ = rgb.shape
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    count = np.asarray(label_file['count'])
    return rgb, t, count

def load_preprocessed_data():

    with open(os.path.join('data//pickles//train//train_x.pkl'), 'rb') as file:
        train_x = pickle.load(file)
    with open(os.path.join('data//pickles//train//train_y.pkl'), 'rb') as file:
        train_y = pickle.load(file)

    with open(os.path.join('data//pickles//val//val_x.pkl'), 'rb') as file:
        val_x = pickle.load(file)
    with open(os.path.join('data//pickles//val//val_y.pkl'), 'rb') as file:
        val_y = pickle.load(file)

    with open(os.path.join('data//pickles//test//test_x.pkl'), 'rb') as file:
        test_x = pickle.load(file)
    with open(os.path.join('data//pickles//test//test_y.pkl'), 'rb') as file:
        test_y = pickle.load(file)

    print(test_y)
    return train_x, train_y, val_x, val_y, test_x, train_y
    

def load_and_process_data():

    root_path = 'data'  # dataset root path
    save_dir = 'data//pickles'

    for phase in ['train', 'val', 'test']:
        x = []
        y = []

        sub_dir = os.path.join(root_path, phase)
        sub_save_dir = os.path.join(save_dir, phase)

        # If save dir doesn't exist make dir
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)

        gt_list = glob(os.path.join(sub_dir, '*json'))

        for gt_path in gt_list:
            name = os.path.basename(gt_path)
            rgb, t, count = generate_data(gt_path)
            gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

            rgbt = np.concatenate([rgb, gray.reshape(gray.shape[0],gray.shape[1],1)], axis=2)
            x.append(rgbt)
            y.append(count)
        

        with open(os.path.join(sub_save_dir, f'{phase}_x.pkl'), 'wb') as file:
            pickle.dump(x, file)
        with open(os.path.join(sub_save_dir, f'{phase}_y.pkl'), 'wb') as file:
            pickle.dump(y, file)
            



load_and_process_data()
load_preprocessed_data()

print("done!")
