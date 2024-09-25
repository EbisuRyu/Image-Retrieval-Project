import chromadb
from tqdm.auto import tqdm
from script.utils import read_image_from_path, get_files_paths, plot_results
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


class ChromadbSearchEngine:
    def __init__(self, metric):
        # Initialize the embedding function and ChromaDB client
        self.__embedding_function = OpenCLIPEmbeddingFunction()
        self.__chroma_client = chromadb.Client()
        # Create or get the collection in ChromaDB based on the specified metric
        self.__collection = self.__get_collection(metric)
        self.files_path = []

    # Private method to get or create a ChromaDB collection based on the metric
    def __get_collection(self, metric):
        if metric == 'l2':
            return self.__chroma_client.get_or_create_collection(
                name="l2_collection",
                metadata={"hnsw_space": "l2"}
            )
        elif metric == 'cosine':
            return self.__chroma_client.get_or_create_collection(
                name="Cosine_collection",
                metadata={"hnsw_space": "cosine"}
            )

    # Private method to get the embedding of a single image using the embedding function
    def __get_single_image_embedding(self, image):
        embedding = self.__embedding_function._encode_image(image)
        return embedding

    # Method to add image embeddings to the ChromaDB collection
    def add_embedding(self, files_path, size):
        ids, embeddings = [], []
        # Loop through all image files and compute their embeddings
        for id_file_path, file_path in tqdm(enumerate(files_path), total=len(files_path)):
            ids.append(f'id_{id_file_path}')
            image = read_image_from_path(file_path, size)
            embedding = self.__get_single_image_embedding(image)
            embeddings.append(embedding)

        # Store the file paths and add the embeddings to the ChromaDB collection
        self.files_path = files_path
        self.__collection.add(
            ids=ids,
            embeddings=embeddings
        )

    # Private method to format the search results from ChromaDB
    def __format_results(self, results):
        # Extract image IDs and convert them to their original index
        ids = [int(id.split('_')[-1]) for id in results['ids'][0]]
        # Map the IDs back to their corresponding file paths
        image_paths = [self.files_path[id] for id in ids]
        # Extract the corresponding scores (distances)
        scores = results['distances'][0]
        return image_paths, scores

    # Method to search for similar images based on a query image
    def search(self, image_path, size, top_k):
        # Read and preprocess the query image
        query_image = read_image_from_path(image_path, size)
        # Get the embedding of the query image
        query_embedding = self.__get_single_image_embedding(query_image)
        # Perform the search in the ChromaDB collection
        results = self.__collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # Format the results to get the image paths and scores
        image_paths, scores = self.__format_results(results)
        # Combine the image paths and scores into a list of tuples
        ls_path_score = list(zip(image_paths, scores))
        return ls_path_score


def main():
    # Path to the root directory containing training images
    data_path = "./data/train/"
    # Get all file paths from the data directory
    files_path = get_files_paths(data_path)
    # Initialize the ChromadbSearchEngine with cosine similarity metric
    chromadb_search_engine = ChromadbSearchEngine(metric='cosine')
    # Add embeddings for all images in the dataset
    chromadb_search_engine.add_embedding(files_path, size=(448, 448))
    # Path to the query image
    query_path = "./data/test/African_crocodile/n01697457_18534.JPEG"
    # Search for the top 6 similar images
    ls_path_score = chromadb_search_engine.search(
        image_path=query_path,
        size=(448, 448),
        top_k=6
    )
    # Display the search results
    print(ls_path_score)
    plot_results(query_path, ls_path_score, top_k=6, reverse=False)


# Entry point of the program
if __name__ == "__main__":
    main()
