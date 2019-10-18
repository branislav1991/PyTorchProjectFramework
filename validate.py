import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os

"""Performs validation of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
"""
def validate(config_file):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    for i, data in enumerate(val_dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

    model.post_epoch_callback(configuration['model_params']['load_checkpoint'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    validate(args.configfile)
