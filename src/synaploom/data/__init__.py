from synaploom.data.features import make_split_feature_stream
from synaploom.data.streams import ContinualStream, TensorTask, iter_tensor_batches
from synaploom.data.synthetic import make_gaussian_cluster_stream

__all__ = [
    "ContinualStream",
    "TensorTask",
    "iter_tensor_batches",
    "make_gaussian_cluster_stream",
    "make_split_feature_stream",
]
