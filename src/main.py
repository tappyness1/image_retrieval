from src.train import train_siamese
from src.data_prep.dataloader import dataloader
import json
import torch
from src.get_embedding import generate_embedding, generate_embeddings
# from src.retrieval import retrieval

class image_retrieval:
    
    def __init__(self):
        return

    def main_function(self, saved_dict_path = None, trained_model_path = None):

        if saved_dict_path:
            f = open(saved_dict_path)
            data_dict = json.load(f)
        else:
            print ("Generating Data Dictionary")
            data_dict = dataloader()

        if trained_model_path:
            trained_network = torch.load(trained_model_path)
        else: 
            print ("Training Network")
            trained_network = train_siamese(data_dict, epochs = 10)

        # change to eval state, but does not seem to be necessary
        # trained_network.eval() 
        print ("Generating Embeddings")
        preds = generate_embeddings(trained_network)
        
        return preds

if __name__ == "__main__":
    model = image_retrieval()
    # model.main_function(saved_dict_path = "data_dict.json", trained_model_path = "trained_model/trained_network.pt")
    model.main_function()
    