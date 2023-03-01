## WHAT YOU NEED TO DO
Build a prototype for similar or reverse image search. Given a query image, your script should quickly retrieve  visually similar photos from a given image collection.
 
## WHAT WE EXPECT
A Python notebook or script with the following components:
1.	Brief description of your approach. Consider detailing what you think might be important for the task. You may find the following points helpful:
    1.	Your algorithm choice
    2.	Evaluating search performance
    3.	Any reference that you found interesting
    4.	Ideas that worked / did not work 
2.	Image collection preparation (ideally, your script should be able to download image collection or its part automatically from the web). Consider using relatively small image collections first
3.	A function for constructing an index for fast retrieval. You are free to use any pre-trained publicly available models or libraries. However, you are not allowed to use free/commercial APIs
4.	A retrieval function that takes a query image and index from Part 3 as arguments and returns K most “similar” images

## BONUS
If you can create various tests to ensure the correctness of your code.
 
## PRIMARY OBJECTIVES
Ensure that the system provides adequate recommendations and in a timely manner.

## Algorithm Candidates

1. SIFT, SURF 
2. ResNet with Triplet Loss
3. Siamese Network with Triplet Loss

Workflow of the Model 

1. Train the model based on the Triple Loss
2. With the model, run it through all the existing images to get the embeddings (encodings/features) - should be NX128 embeddings (the index)
3. During inference, the model will be run on the query image, and compare it with the pre-calculated embeddings - get similarity scores
4. Rank the similarity score and get the K most similar images
5. Visualise the images



Codes used:  
https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch/notebook#Define-Neural-Network  
https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html  

Dataset used:
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
