"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, configuration).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.

In the function <__init__>, you need to define four lists:
    -- self.network_names (str list):       define networks used in our training.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them.
"""

import importlib
from models.base_model import BaseModel
from torch.optim import lr_scheduler


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

        In the file, the class called DatasetNameModel() will
        be instantiated. It has to be a subclass of BaseModel,
        and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def create_model(configuration):
    """Create a model given the configuration.

    This is the main interface between this package and train.py/validate.py
    """
    model = find_model_using_name(configuration['model_name'])
    instance = model(configuration)
    print("model [{0}] was created".format(type(instance).__name__))
    return instance