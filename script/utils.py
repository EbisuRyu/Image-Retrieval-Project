import math
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read_image_from_path(path, size):
    image = Image.open(path).convert('RGB').resize(size)
    return np.array(image)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path


def show_images(query_path, result_image_paths):
    image_paths = [query_path] + result_image_paths
    columns = int(math.sqrt(len(image_paths)))
    rows = int(np.ceil(len(image_paths) / columns))
    fig = plt.figure(figsize=(15, 10))

    query_image = plt.imread(query_path)
    ax = fig.add_subplot(rows, columns, 1)
    ax.set_title(f"Query Image: {query_path.split('/')[-2]}")
    plt.imshow(query_image)
    plt.axis('off')

    for i in range(2, len(image_paths) + 1):
        image = plt.imread(image_paths[i - 1])
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(f"Top {i - 1}: {image_paths[i - 1].split('/')[-2]}")

        plt.imshow(image)
        plt.axis('off')
    plt.show()


def plot_results(query_path, ls_path_score, top_k=10, reverse=False):
    sorted_ls_path_score = sorted(
        ls_path_score, key=lambda item: item[1], reverse=reverse)
    result_image_paths = [path for path, _ in sorted_ls_path_score[:top_k+1]]
    show_images(query_path, result_image_paths)


def get_files_paths(path):
    files_path = []
    class_name = sorted(os.listdir(path))
    for label in class_name:
        label_path = path + '/' + label
        file_names = os.listdir(label_path)
        for file_name in file_names:
            file_path = label_path + '/' + file_name
            files_path.append(file_path)
    return files_path
