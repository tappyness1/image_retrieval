import cv2
import torch
import os
import numpy as np
import json

def generate_embedding(model, img_path = "data/jpg/image_00001.jpg", hw = 64):
    """specify the image path, then generate embeddings for that particular image

    Args:
        model (_type_): _description_
        img_path (str, optional): _description_. Defaults to "data/jpg/image_00001.jpg".
        hw (int, optional): _description_. Defaults to 64.
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(hw, hw), interpolation=cv2.INTER_CUBIC).reshape((3,hw,hw))
    img = torch.tensor(np.float32(img))
    pred = model(img)

    return pred

def generate_embeddings(model, folder = "data/jpg", stored_output_path = "stored_embeddings/embeddings.json"):
    # using the saved model weights, generate the embeddings of the images
    # return dictionary of index to be saved
    all_img = os.listdir(folder)
    embeddings_dict = {img: generate_embedding(model, f"{folder}/{img}").detach().numpy().ravel().tolist() for img in all_img if img[-4:] in ['.jpg', '.png']}
    
    if stored_output_path: 

        with open(stored_output_path, "w") as fp:
            json.dump(embeddings_dict,fp) 

    return embeddings_dict