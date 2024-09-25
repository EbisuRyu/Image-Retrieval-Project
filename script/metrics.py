import numpy as np

def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(query - data), axis=axis_batch_size)

def mean_squared_error(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean(np.square(query - data), axis=axis_batch_size)

def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query ** 2))
    data_norm = np.sqrt(np.sum(data ** 2, axis=axis_batch_size))
    return np.sum(query * data, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(np.float32).eps)

def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query_mean ** 2))
    data_norm = np.sqrt(np.sum(data_mean ** 2, axis=axis_batch_size))
    return np.sum(query_mean * data_mean, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(np.float32).eps)