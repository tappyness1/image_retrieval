from src.train import train_siamese
from src.data_prep.dataloader import dataloader
import json
import torch
from src.get_embedding import generate_embedding, generate_embeddings
from src.retrieval import golden_retriever

class image_retrieval:
    
    def __init__(self):
        return

    def main_function(self, saved_dict_path = None, trained_model_path = None, stored_embeddings_path = None):

        if saved_dict_path:
            f = open(saved_dict_path)
            self.data_dict = json.load(f)
        else:
            print ("Generating Data Dictionary")
            self.data_dict = dataloader()

        if trained_model_path:
            self.trained_network = torch.load(trained_model_path)
        else: 
            print ("Training Network")
            self.trained_network = train_siamese(self.data_dict, epochs = 10)

        # change to eval state, but does not seem to be necessary
        # trained_network.eval() 
        
        if stored_embeddings_path:
            f = open(stored_embeddings_path)
            self.embeddings = json.load(f)
        else: 
            print ("Generating Embeddings")
            self.embeddings = generate_embeddings(self.trained_network)

        return
    
    def retrieve_similar_img(self, input_image):
        retrieval_obj = golden_retriever(self.trained_network, self.embeddings)
        retrieval_obj.get_euclidean_dist(input_image) 
        img_list = retrieval_obj.retrieval()
        return img_list

if __name__ == "__main__":
    model = image_retrieval()
    model.main_function(saved_dict_path = "data_dict.json", trained_model_path = "trained_model/trained_network.pt", stored_embeddings_path = "stored_embeddings/embeddings.json")
    # model.main_function(saved_dict_path = "data_dict.json")
    print (model.retrieve_similar_img("data/jpg/image_00001.jpg"))
    