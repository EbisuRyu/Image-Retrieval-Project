import os
import numpy as np
from script.search_engine.wrapper import SearchEngineWrapper
from script.utils import read_image_from_path, folder_to_images, plot_results
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


class CLIPSearchEngine(SearchEngineWrapper):
    def __init__(self, root_image_path):
        super().__init__(root_image_path)
        # Initialize the embedding function using OpenCLIP
        self.__embedding_function = OpenCLIPEmbeddingFunction()

    def __get_single_image_embedding(self, image):
        # Encode the image to get its embedding and convert it to a numpy array
        embedding = self.__embedding_function._encode_image(image)
        return np.array(embedding)

    # Method to compute the score between the query image and all images in the root directory
    def get_score(self, query_path, metric, size):
        # Read and preprocess the query image
        query_image = read_image_from_path(query_path, size)
        # Get the embedding of the query image
        query_embedding = self.__get_single_image_embedding(query_image)
        # Retrieve the appropriate metric function based on the metric name
        metric_function = self._get_metric_function(metric)
        # List to store tuples of (image path, score)
        ls_path_score = []

        for folder in os.listdir(self.root_image_path):
            embedding_list = []
            # Only process folders that are in the _class_name list
            if folder in self._class_name:
                path = os.path.join(self.root_image_path, folder)
                images_np, images_path = folder_to_images(path, size)
                # Compute embeddings for each image in the folder
                for image_np in images_np:
                    embedding = self.__get_single_image_embedding(
                        image=image_np.astype(np.uint8)
                    )
                    embedding_list.append(embedding)
                # Compute the scores between the query embedding and the embeddings in the folder
                scores = metric_function(
                    query=query_embedding,
                    data=np.stack(embedding_list)
                )
                ls_path_score.extend(zip(images_path, scores))
        return ls_path_score


def main():
    # Path to the root directory containing training images
    root_image_path = "./data/train/"
    # Initialize the CLIPSearchEngine object
    clip_search_engine = CLIPSearchEngine(root_image_path)
    # Path to the query image
    query_path = "./data/test/African_crocodile/n01697457_18534.JPEG"
    # Compute the scores between the query image and all images in the database
    ls_path_score = clip_search_engine.get_score(
        query_path=query_path,
        metric='l1',
        size=(448, 448)
    )
    # Display the search results (top 6 closest images)
    plot_results(
        query_path=query_path,
        ls_path_score=ls_path_score,
        top_k=6,
        reverse=False
    )


# Entry point of the program
if __name__ == "__main__":
    main()
