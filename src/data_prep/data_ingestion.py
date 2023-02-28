from src.data_prep.dataloader import dataloader
import cv2
import torch
import numpy as np

def prep_img(img_string, hw):
    img = cv2.imread("data/jpg/image_" + img_string + ".jpg")
    out = cv2.resize(img, dsize=(hw, hw), interpolation=cv2.INTER_CUBIC).reshape((3,hw,hw))
    out = torch.tensor(np.float32(out))/255

    return out

def ingest_data(anchor, positive, negative, hw = 224):

    anchor_out = prep_img(anchor, hw)
    positive_out = prep_img(positive, hw)
    negative_out = prep_img(negative, hw)

    return anchor_out, positive_out, negative_out 
