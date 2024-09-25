import os
from script.search_engine.wrapper import SearchEngineWrapper
from script.utils import read_image_from_path, folder_to_images, plot_results


class BasicSearchEngine(SearchEngineWrapper):
    def __init__(self, root_image_path):
        super().__init__(root_image_path)

    # Method to compute the score between the query image and all images in the root directory
    def get_score(self, query_path, metric, size):
        # Read the query image from the specified path with the given size
        query_image = read_image_from_path(query_path, size)
        # Retrieve the appropriate metric function based on the metric name
        metric_function = self._get_metric_function(metric)
        # List to store tuples of (image path, score)
        ls_path_score = []
        for folder in os.listdir(self.root_image_path):
            # Only process folders that are in the _class_name list
            if folder in self._class_name:
                path = os.path.join(self.root_image_path, folder)
                images_np, images_path = folder_to_images(path, size)
                # Compute the scores between the query image and the images in the folder
                scores = metric_function(query_image, images_np)
                ls_path_score.extend(list(zip(images_path, scores)))

        return ls_path_score


def main():
    # Path to the root directory containing training images
    root_image_path = "./data/train/"
    # Initialize the BasicSearchEngine object
    basic_search_engine = BasicSearchEngine(root_image_path)
    # Path to the query image
    query_path = "./data/test/African_crocodile/n01697457_18534.JPEG"
    # Compute the scores between the query image and all images in the database
    ls_path_score = basic_search_engine.get_score(query_path, 'l1', size=(448, 448))
    # Display the search results (top 6 closest images)
    plot_results(query_path, ls_path_score, top_k=6, reverse=False)


# Entry point of the program
if __name__ == "__main__":
    main()
