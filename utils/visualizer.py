import numpy as np
import sys
from subprocess import Popen, PIPE
import utils
import visdom


class Visualizer():
    """This class includes several functions that can display images and print logging information.
    """

    def __init__(self, configuration):
        """Initialize the Visualizer class.

        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.display_id = 0
        self.name = configuration['name']

        self.ncols = 0
        self.vis = visdom.Visdom()
        if not self.vis.check_connection():
            self.create_visdom_connections()


    def reset(self):
        """Reset the visualization.
        """
        pass


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at the default port.
        """
        cmd = sys.executable + ' -m visdom.server'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Display the current losses on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
            losses: Training losses stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([losses[k] for k in self.loss_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.loss_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.loss_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.create_visdom_connections()


    def plot_current_validation_metrics(self, epoch, metrics):
        """Display the current validation metrics on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            losses: Validation metrics stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.val_plot_data['X'].append(epoch)
        self.val_plot_data['Y'].append([metrics[k] for k in self.val_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.val_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id+1)
        except ConnectionError:
            self.create_visdom_connections()


    def plot_roc_curve(self, fpr, tpr, thresholds):
        """Display the ROC curve.

        Input params:
            fpr: False positive rate (1 - specificity).
            tpr: True positive rate (sensitivity).
            thresholds: Thresholds for the curve.
        """
        try:
            self.vis.line(
                X=fpr,
                Y=tpr,
                opts={
                    'title': 'ROC Curve',
                    'xlabel': '1 - specificity',
                    'ylabel': 'sensitivity',
                    'fillarea': True},
                win=self.display_id+2)
        except ConnectionError:
            self.create_visdom_connections()


    def show_validation_images(self, images):
        """Display validation images. The images have to be in the form of a tensor with
        [(image, label, prediction), (image, label, prediction), ...] in the 0-th dimension.
        """
        # zip the images together so that always the image is followed by label is followed by prediction
        images = images.permute(1,0,2,3)
        images = images.reshape((images.shape[0]*images.shape[1],images.shape[2],images.shape[3]))

        # add a channel dimension to the tensor since the excepted format by visdom is (B,C,H,W)
        images = images[:,None,:,:]

        try:
            self.vis.images(images, win=self.display_id+3, nrow=3)
        except ConnectionError:
            self.create_visdom_connections()


    def print_current_losses(self, epoch, max_epochs, iter, max_iters, losses):
        """Print current losses on console.

        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
        message = '[epoch: {}/{}, iter: {}/{}] '.format(epoch, max_epochs, iter, max_iters)
        for k, v in losses.items():
            message += '{0}: {1:.6f} '.format(k, v)

        print(message)  # print the message
