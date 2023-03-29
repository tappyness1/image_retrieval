# POC - Image Query algorithm

by Calvin Neo

## TOC

1. Algorithm Candidates
1. Workflow of the Model
1. Getting the Data
1. General Training and Embedding Generation
1. Retrieval
1. Evaluation
1. Results of Experiments
1. Future Works

## Algorithm Candidates
 
1. ResNet with Triplet Loss
1. Siamese Network with Triplet Loss
1. SIFT, SURF (not explored)

Workflow of the Model 

1. Get the Data
1. Train the model based on the Triple Loss (`train.py`)
1. With the model, run it through all the existing images to get the embeddings (encodings/features) - should be NX128 embeddings (the index) (`get_embeddings.py`)
1. During inference, the model will be run on the query image, and compare it with the pre-calculated embeddings - get similarity scores (`retrieval.py`)
1. Rank the similarity score and get the K most similar images (`retrieval.py`)
1. Model Evaluation (`evaluation.py`)
1. Visualise the images (`evaluation.py`)

## Getting the data

The dataset "102 Flowers" was used when building the model. To get the dataset, visit - https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

The relevant files are "Dataset Images" and "The image labels"

The script - src/data_prep/get_data.py also handles the download. 

What it does is:

1. Download the image dataset into /data
1. Unpacks the dataset into /data/jpg it will create jpg folder
1. Downloads the image labels into the jpg folder (currently not usable)

However, it is currently bugged in the last step as it is unable to download the image labels. Thus, after running the script, manually download and insert the image labels into jpg folder. 

To run the code (until step 2) - 

```
# make sure you are root folder
python -m src.data_prep.get_data
```

## General Training and Embedding Generation

The `main.py` script handles the model training, embedding generation and model evaluation parts of the POC. 

### Configuration 

Below is a reference to the configuration file which will be referred to extensively in subsequent subsections.

```
general:
  labels_fpath: 'data/jpg/imagelabels.mat'

train:
  epochs: 10
  hw: 224
  save_network_path: "trained_model/trained_network_model4_resnet_freeze_10epochs.pt"
  embeddings_output_path: "stored_embeddings/embeddings_model3_resnet_freeze_10epochs.json"
  network: "resnet"

resnet:
  freeze_backbone: True # if True, only the final fully connected layer (weights on the 128 output embeddings) will be trained
```

### Training the Model

The main training script is handeld in `train.py`

To start with, modify the training configurations. 

For example, if you wish to train the siamese network, 100 epochs, image height and width at 64x64, then the training parameters would be 

```
general:
  labels_fpath: 'data/jpg/imagelabels.mat' # not to be touched

train:
  epochs: 100
  hw: 64
  save_network_path: "trained_model/trained_network_<customise your network filename here>.pt"
  embeddings_output_path: "stored_embeddings/embeddings_<customise your embeddings filename here>.json"
  network: "siamese" # the only relevant string here is "resnet". Everything else defaults to Siamese

resnet:
  freeze_backbone: True # not relevant
```

Once all sorted, modify `main.py` with the following:

```
if __name__ == "__main__":
    model = image_retrieval()
    model.main_function(saved_dict_path = "data_dict.json")
```

Then run using the command

```
# make sure you are root folder
python -m src.main
```

Training will begin, and will be faster if you have GPU. 

## Embeddings

Embeddings will be generated and stored based on your configuration file.

## Retrieval 

Running the retrival method is the same as running the `main.py`. 

Here, assuming that training and embedding generation has been completed, the configuration for the model and embeddings path changed as follows:

```
...

train:
  epochs: 10
  hw: 64
  save_network_path: "trained_model/trained_network_model1.pt"
  embeddings_output_path: "stored_embeddings/embeddings_model1.json"
  network: "siamese"
...
```

Now assuming you want to perform the search on image_00001.jpg, then put the following into `main.py`

```
if __name__ == "__main__":
    model = image_retrieval()
    model.main_function(saved_dict_path = "data_dict.json", trained_model_path = "trained_model/trained_network_model1.pt", stored_embeddings_path = "stored_embeddings/embeddings_model1.json")
    model.retrieve_similar_img(input_image = "data/jpg/image_00001.jpg")
```

Then run using the command

```
# make sure you are root folder
python -m src.main
```

## Evaluation and visualisation

Evaluation is handled in evaluation.py

Evaluation of the models follows the example of "If given an image of Bob, return images of Bob instead of Tom or Sally"

Hence, classification metrics was used. In particular, this POC used F1 Score as the harmonic mean between Precision and Recall to evaluate the models.

More information of the results of experiments can be found in "Results of Experiments"

Evaluation also includes visualising the query results. It always outputs the top 10 results.

For a demonstration of how to run evaluation as well as the visualisation, refer to `final_report.ipynb`

## Results of Experiments

Results of Experiments can be viewed in the "final_report.ipynb"

Summary:

| Experiment 	| Details                                                           	| F1 Score 	| Query Time 	|
|------------	|-------------------------------------------------------------------	|----------	|------------	|
| 1          	| Siamese Network, 64x64x3 images training data, 10 epochs          	| 0.19     	| 2.6s       	|
| 2          	| Siamese Network, 224x224x3 images training data, 10 epochs        	| 0.07     	| 2.5s       	|
| 3          	| ResNet50 1 epoch, 224x224x3 images                                	| 0.11     	| 1.8s       	|
| 4          	| ResNet50 (Pretrained ImageNet) 1 epoch, 224x224x3 images          	| 0.11     	| 2.3s       	|
| 5          	| ResNet50 (Pretrained ImageNet, Frozen) 10 epoch, 224x224x3 images 	| 0.11     	| 2.1s       	|

Findings

1. Of the experiments, ResNet, trained on ImageNet dataset, was expected to perform the best. However, this was not the case. A simple Siamese network modified to feed in 3 channels images was sufficient
2. Images with larger size was expected to perform better as there would be more features to extract. This was not the case, as 64x64 images in Siamese network produced the best results.
3. Experiment 1 was able to get more similar images than the rest of the network, but it did not necessarily get the exact image back as its top search.
4. ResNet models alwasy gets the exact image back as its top search result, but subsequent searches do not yield the right image class. 

## Future Works 

Some potential areas to explore if given more time and resources: 

### Making Model Faster
1. Attempt PCA to reduce the dimension of embeddings 
1. Use Kmeans Clustering such that the retrieval only needs to search for relevant clusters and retrieve embeddings to compare distance instead of all the embeddings

### Increase Model Precision
1. Use different backbone (ImageNet may not have been the best to use here)
1. Put in tests to ensure that the images are being fed properly (OpenCV vs PyTorch issues)
1. Try out different architectures

Codes adapted from:  
https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch/notebook#Define-Neural-Network  
https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html  

Dataset used:
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
