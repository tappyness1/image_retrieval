import cv2
import torch
import os
import numpy as np
import json

def generate_embedding(model, img_path = "data/jpg/image_00001.jpg", hw = 64, network = 'siamese'):
    """specify the image path, then generate embeddings for that particular image

    Args:
        model (_type_): _description_
        img_path (str, optional): _description_. Defaults to "data/jpg/image_00001.jpg".
        hw (int, optional): _description_. Defaults to 64.
    """
 
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(hw, hw), interpolation=cv2.INTER_CUBIC).reshape((3,hw,hw))
    img = torch.tensor(np.float32(img))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = img.to(device)
    if network == 'resnet':
        pred = model(img.reshape(1,3,hw,hw))
    else:
        pred = model(img)

    return pred

def generate_embeddings(model, folder = "data/jpg", embeddings_output_path = "stored_embeddings/embeddings.json", hw = 64, network = 'siamese'):
    # using the saved model weights, generate the embeddings of the images
    # return dictionary of index to be saved
    all_img = os.listdir(folder)
    embeddings_dict = {img: generate_embedding(model, f"{folder}/{img}", hw = hw, network = network).detach().cpu().numpy().ravel().tolist() for img in all_img if img[-4:] in ['.jpg', '.png']}
    
    if embeddings_output_path: 
        # print(f"saving emdeddings to {embeddings_output_path}")
        with open(embeddings_output_path, "w") as fp:
            json.dump(embeddings_dict,fp) 

    return embeddings_dict