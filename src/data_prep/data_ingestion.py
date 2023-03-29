from src.data_prep.dataloader import dataloader
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, data_dict, hw=224):

        self.train, _, _ = data_dict
        self.hw = hw

    def __len__(self):
        return len(list(self.train.keys()))

    def __getitem__(self, idx):
        idx_key = self.train[list(self.train.keys())[idx]]
        anchor = idx_key["anchor"]
        positive = idx_key["positive"]
        negative = idx_key["negative"]
        anchor_out, positive_out, negative_out = ingest_data(
            anchor, positive, negative, hw=self.hw
        )
        return anchor_out, positive_out, negative_out


def prep_img(img_string, hw):
    img = cv2.imread("data/jpg/image_" + img_string + ".jpg")
    out = cv2.resize(img, dsize=(hw, hw), interpolation=cv2.INTER_CUBIC).reshape(
        (3, hw, hw)
    )
    out = torch.tensor(np.float32(out)) / 255

    return out


def ingest_data(anchor, positive, negative, hw=224):

    anchor_out = prep_img(anchor, hw)
    positive_out = prep_img(positive, hw)
    negative_out = prep_img(negative, hw)

    return anchor_out, positive_out, negative_out
