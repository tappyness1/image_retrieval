from src.get_embedding import generate_embedding
import numpy as np

class golden_retriever:

    def __init__(self, model, stored_embeddings):
        self.index = {}
        self.model = model
        self.stored_embeddings = stored_embeddings
        return

    def get_euclidean_dist(self, input_image):

        input_embeddings = generate_embedding(self.model, input_image)
        input_embeddings = input_embeddings.detach().numpy()
        curr_embeddings = []
        ind = 0
        for img, embeddings in self.stored_embeddings.items():
            curr_embeddings.append(embeddings)
            self.index[str(ind)] = img
            ind += 1
        curr_embeddings = np.array(curr_embeddings)
        # curr_embeddings = np.array([embeddings for _, embeddings in self.stored_embeddings.items()])
        self.dist = np.sqrt(((curr_embeddings - input_embeddings)**2).sum(axis = 1))
        print (self.dist.max())

    def retrieval(self, k = 10, threshold = 0.2):
        """retrieves the index of top k results and returns their image file name

        Args:
            k (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        idx = np.argsort(self.dist)[:k]
        imglist = [self.index[str(i)] for i in idx]
        return imglist