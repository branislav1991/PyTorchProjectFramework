"""This package includes all the modules related to data loading and preprocessing.

    To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
"""
import importlib
from torch.utils import data
from datasets.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError('In {0}.py, there should be a subclass of BaseDataset with class name that matches {1} in lowercase.'.format(dataset_filename, target_dataset_name))

    return dataset


def create_dataset(configuration):
    """Create a dataset given the configuration (loaded from the json file).

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and train.py/validate.py

    Example:
        from datasets import create_dataset
        dataset = create_dataset(configuration)
    """
    data_loader = CustomDatasetDataLoader(configuration)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading
        according to the configuration.
    """
    def __init__(self, configuration):
        self.configuration = configuration
        dataset_class = find_dataset_using_name(configuration['dataset_name'])
        self.dataset = dataset_class(configuration)
        print("dataset [{0}] was created".format(type(self.dataset).__name__))

        # if we use custom collation, define it as a staticmethod in the dataset class
        custom_collate_fn = getattr(self.dataset, "collate_fn", None)
        if callable(custom_collate_fn):
            self.dataloader = data.DataLoader(self.dataset, **configuration['loader_params'], collate_fn=custom_collate_fn)
        else:
            self.dataloader = data.DataLoader(self.dataset, **configuration['loader_params'])


    def load_data(self):
        return self


    def get_custom_dataloader(self, custom_configuration):
        """Get a custom dataloader (e.g. for exporting the model).
            This dataloader may use different configurations than the
            default train_dataloader and val_dataloader.
        """
        custom_collate_fn = getattr(self.dataset, "collate_fn", None)
        if callable(custom_collate_fn):
            custom_dataloader = data.DataLoader(self.dataset, **self.configuration['loader_params'], collate_fn=custom_collate_fn)
        else:
            custom_dataloader = data.DataLoader(self.dataset, **self.configuration['loader_params'])
        return custom_dataloader
    

    def __len__(self):
        """Return the number of data in the dataset.
        """
        return len(self.dataset)


    def __iter__(self):
        """Return a batch of data.
        """
        for data in self.dataloader:
            yield data
