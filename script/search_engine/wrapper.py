import os
from script.metrics import absolute_difference, mean_squared_error, cosine_similarity, correlation_coefficient


class SearchEngineWrapper:
    def __init__(self, root_image_path):
        self.root_image_path = root_image_path
        self._class_name = sorted(list(os.listdir(root_image_path)))

    def _get_metric_function(self, metric):
        if metric == 'l1':
            return absolute_difference
        elif metric == 'l2':
            return mean_squared_error
        elif metric == 'cosine':
            return cosine_similarity
        elif metric == 'correlation':
            return correlation_coefficient
        else:
            raise ValueError(f"Invalid metric: {metric}")
