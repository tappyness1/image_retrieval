from torch.nn import TripletMarginLoss
from src.model import SiameseNetwork
from torch import randn, optim
import torch
from src.data_prep.data_ingestion import ingest_data
import time
from tqdm import tqdm


def train_siamese(data_dict, subset = 0, epochs = 100, save_network_path = "trained_model/trained_network.pt", save = True):

    triplet_loss = TripletMarginLoss()
    # test input with a 224x224x3 img
    train, validation, test =  data_dict
    
    all_dict_keys_len = len(list(train.keys()))
    network = SiameseNetwork()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # print (device)
    network.to(device)
    # start = time.time()
    # add tqdm here
    for epoch in range(epochs):
        print (f"Epoch {epoch + 1}:")
        for i in tqdm(range(all_dict_keys_len - subset)):
            optimizer.zero_grad() 

            # todo - refactor below (should not have to call these three times)
            anchor = train[list(train.keys())[i]]['anchor']
            positive = train[list(train.keys())[i]]['positive']
            negative = train[list(train.keys())[i]]['negative']
            
            anchor_input, positive_input, negative_input = ingest_data(anchor, positive, negative, 64)
        
            anchor_output = network.forward(anchor_input.to(device))
            positive_output = network.forward(positive_input.to(device))
            negative_output = network.forward(negative_input.to(device))

            loss = triplet_loss(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()
    
    print("training done")
    # print (f"Time Taken: {time.time() - start}")

    if save:
        torch.save(network, save_network_path)

    return network

if __name__ == "__main__":
    train_siamese()