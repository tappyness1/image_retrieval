from src.train import train_siamese, train_resnetembedder
from src.data_prep.dataloader import dataloader
import json
import torch
from src.get_embedding import generate_embedding, generate_embeddings
from src.retrieval import golden_retriever
from src.evaluation import get_tpfp, get_image_labels, visualise
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import yaml

class image_retrieval:
    
    def __init__(self, cfg_path = "cfg/cfg.yml"):
        cfg_file = open(cfg_path)
        self.cfg_obj = yaml.load(cfg_file, Loader=yaml.FullLoader)
        self.network = self.cfg_obj['train']['network']
        self.hw = self.cfg_obj['train']['hw']
        return

    def main_function(self, saved_dict_path = None, trained_model_path = None, stored_embeddings_path = None):
        
        # self.hw = self.cfg_obj['train']['hw']
        epochs = self.cfg_obj['train']['epochs']
        save_network_path = self.cfg_obj['train']['save_network_path']
        labels_fpath = self.cfg_obj['general']['labels_fpath']
        embeddings_output_path = self.cfg_obj['train']['embeddings_output_path']

        if saved_dict_path:
            f = open(saved_dict_path)
            self.data_dict = json.load(f)
        else:
            print ("Generating Data Dictionary")
            self.data_dict = dataloader(labels_fpath = labels_fpath)

        if trained_model_path:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.trained_network = torch.load(trained_model_path, map_location=torch.device(device))

        else: 
            print ("Training Network")
            if self.network == 'resnet':
                print("Training Resnet")
                freeze_backbone = self.cfg_obj['resnet']['freeze_backbone']
                self.trained_network = train_resnetembedder(self.data_dict,save_network_path = save_network_path, epochs = epochs, hw = self.hw, freeze_backbone = freeze_backbone)
            else:
                self.trained_network = train_siamese(self.data_dict,save_network_path = save_network_path, epochs = epochs, hw = self.hw)
        # else: 
        #     print ("Training Network")

        #     print(f"Training {self.network}")
        #     freeze_backbone = self.cfg_obj['resnet']['freeze_backbone']
        #     self.trained_network = train(self.data_dict, network = self.network, save_network_path = save_network_path, epochs = epochs, hw = self.hw, freeze_backbone = freeze_backbone)

        # change to eval state, but does not seem to be necessary
        # trained_network.eval() 
        
        if stored_embeddings_path:
            f = open(stored_embeddings_path)
            self.embeddings = json.load(f)
        else: 
            print ("Generating Embeddings")
            self.embeddings = generate_embeddings(self.trained_network, embeddings_output_path = embeddings_output_path, hw = self.hw, network = self.network)

        return
    
    # consider turning below into helper function
    def retrieve_similar_img(self, input_image = "data/jpg/image_00001.jpg", gen_visual = False):
        """_summary_

        Args:
            input_image (_type_): _description_

        Returns:
            _type_: _description_
        """
        retrieval_obj = golden_retriever(self.trained_network, self.embeddings, self.network)
        retrieval_obj.get_euclidean_dist(input_image, hw = self.hw) 
        img_list = retrieval_obj.retrieval()
        if gen_visual:
            visualise(input_image, img_list)
        return img_list

    def evaluate(self, labels_fpath='data/jpg/imagelabels.mat'):
        # get anchors from validation data dictionary
        val_set = self.data_dict[1]
        val_anchors = [v['anchor'] for _, v in val_set.items()]
        img_label_dict = get_image_labels(labels_fpath)
        df_full = pd.DataFrame()
        
        for anchor in val_anchors:
            path_to_anchor = f"data/jpg/image_{anchor}.jpg"
            img_list = self.retrieve_similar_img(path_to_anchor)
            df = get_tpfp(anchor, img_list, img_label_dict)
            df_full = pd.concat([df_full, df])

        cm = confusion_matrix(df_full['Ground Truth'], df_full['Prediction'])
        cr = classification_report(df_full['Ground Truth'], df_full['Prediction'])
        return cm, cr


if __name__ == "__main__":
    model = image_retrieval()
    # model.main_function(saved_dict_path = "data_dict.json", trained_model_path = "trained_model/trained_network.pt", stored_embeddings_path = "stored_embeddings/embeddings.json")
    # model.main_function(saved_dict_path = "data_dict.json", trained_model_path = "trained_model/trained_network_model3_resnet.pt", stored_embeddings_path = "stored_embeddings/embeddings_model3_resnet.json")
    # model.main_function(saved_dict_path = "data_dict.json", trained_model_path = "trained_model/trained_network_model4_resnet_pretrained.pt", stored_embeddings_path = "stored_embeddings/embeddings_model4_resnet_pretrained.json")
    # model.main_function(saved_dict_path = "data_dict.json", trained_model_path = "trained_model/trained_network_model5_resnet_freeze_10epochs.pt", stored_embeddings_path = "stored_embeddings/embeddings_model5_resnet_freeze_10epochs.json")
    model.main_function(saved_dict_path = "data_dict.json")
    # model.retrieve_similar_img(input_image = "data/jpg/image_00001.jpg")
    