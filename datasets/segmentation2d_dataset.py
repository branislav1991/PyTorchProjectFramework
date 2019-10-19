from datasets.base_dataset import get_transform
from datasets.base_dataset import BaseDataset
import torch


class Segmentation2DDataset(BaseDataset):
    """Represents a 2D segmentation dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)


    def __getitem__(self, index):
        # get source image as x
        # get labels as y
        return (x, y)

    def __len__(self):
        # return the size of the dataset
        return 1
