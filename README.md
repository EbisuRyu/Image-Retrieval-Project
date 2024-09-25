# Image Retrieval System

This project showcases the development of an image retrieval system designed to return relevant images from a database based on a provided query image. The system offers both basic and advanced retrieval functionalities to efficiently locate similar images.

## Key Components
- **Basic Image Retrieval System**: A simple image search system using traditional similarity metrics.
- **Advanced Image Retrieval System**: Utilizes the CLIP model and a vector database for improved retrieval accuracy and performance.

## Basic Image Retrieval System
This module implements a straightforward image retrieval system that allows users to find and retrieve images similar to a query image. Several similarity metrics are used to evaluate how closely two images resemble each other, including:
- L1 distance
- L2 distance
- Cosine similarity
- Correlation coefficient

### Dataset Structure:
- `data/train`: Contains the images used as the database for retrieval.
- `data/test`: Contains the query images to search against the database.

## Advanced Image Retrieval System
The advanced system builds on the basic version by incorporating deep learning techniques, specifically using the CLIP model to extract robust feature vectors from images. This significantly enhances the accuracy of the retrieval process.

### How it Works:
Pretrained deep learning models are leveraged to search and retrieve relevant images based on their content. These models, trained on vast datasets, capture essential visual features, making them highly effective for content-based image retrieval.

When a query image is submitted, the model computes a feature vector for it, which is then compared to the precomputed vectors of the database images. The system identifies and returns the most similar images based on these feature comparisons. These models are capable of recognizing complex image attributes such as texture, shape, and color, ensuring precise and efficient retrieval.

## Installation

To set up the environment, follow these steps:

1. Create and activate a conda environment:
   ```bash
   conda create -n image-retrieval-env python
   conda activate image-retrieval-env
   ```

2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

## Instructions

- `search_engine/basic_search_engine.py`: Contains the source code for the basic image retrieval system using traditional similarity metrics.
- `search_engine/clip_search_engine.py`: Contains the source code for the advanced retrieval system using the CLIP model.
- `search_engine/chroma_search_engine.py`: Implements the advanced system with both the CLIP model and a Chroma vector database for enhanced retrieval capabilities.

This project combines simplicity with advanced techniques to deliver a flexible and efficient image retrieval solution.