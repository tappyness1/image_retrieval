from src.data_prep.dataloader import dataloader
import cv2
import torch
import numpy as np

def ingest_data(anchor, positive, negative, hw = 224):
    anchor_img = cv2.imread("data/jpg/image_" + anchor + ".jpg")
    positive_img = cv2.imread("data/jpg/image_" + positive + ".jpg")
    negative_img = cv2.imread("data/jpg/image_" + negative + ".jpg")

    anchor_out = cv2.resize(anchor_img, dsize=(hw, hw), interpolation=cv2.INTER_CUBIC).reshape((3,hw,hw ))
    positive_out = cv2.resize(positive_img, dsize=(hw, hw), interpolation=cv2.INTER_CUBIC).reshape((3,hw,hw ))
    negative_out = cv2.resize(negative_img, dsize=(hw, hw), interpolation=cv2.INTER_CUBIC).reshape((3,hw,hw ))

    anchor_out = torch.tensor(np.float32(anchor_out))
    positive_out = torch.tensor(np.float32(positive_out))
    negative_out = torch.tensor(np.float32(negative_out))

    return anchor_out, positive_out, negative_out 
