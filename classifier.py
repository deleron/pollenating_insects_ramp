import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum, rmsprop, adagrad
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np
from caffezoo.googlenet import GoogleNet
from itertools import repeat
from sklearn.pipeline import make_pipeline

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2).astype(np.float32)

def apply_filter(x, filt):
    return np.array([ ex * filt for ex in x])

def sample_from_rotation_x(x):
    x_extends = []
    for i in range(x.shape[0]):
        x_extends.extend([
        np.array([x[i,:,:,0], x[i,:,:,1], x[i,:,:,2]]),
        np.array([np.rot90(x[i,:,:,0]),np.rot90(x[i,:,:,1]), np.rot90(x[i,:,:,2])]),
        np.array([np.rot90(x[i,:,:,0],2),np.rot90(x[i,:,:,1],2), np.rot90(x[i,:,:,2],2)]),
        np.array([np.rot90(x[i,:,:,0],3),np.rot90(x[i,:,:,1],3), np.rot90(x[i,:,:,2],3)])
        ])
    x_extends = np.array(x_extends) #.transpose((0, 2, 3, 1))
    return x_extends

def sample_from_rotation_y(y):
    y_extends = []
    for i in y:
        y_extends.extend( repeat( i ,4) )
    return np.array(y_extends)

class FlipBatchIterator(BatchIterator):    
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 4, replace=False)
        Xb[indices] = Xb[indices, :, ::-1]
        
        # Drop randomly half of the features in each batch:
        bf = Xb.shape[2]
        indices_features = np.random.choice(bf, bf / 2, replace=False)
        Xb = Xb.transpose((2, 0, 1, 3))
        Xb[indices_features] = Xb[indices_features]
        Xb = Xb.transpose((1, 2, 0, 3))
        return Xb, yb

def build_model(crop_value):    
    L=[
       (layers.InputLayer, {'shape':(None, 3, 64-2*crop_value, 64-2*crop_value)}),
       (layers.Conv2DLayer, {'num_filters':64, 'filter_size':(3,3), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (2, 2)}),
       (layers.Conv2DLayer, {'num_filters':128, 'filter_size':(2,2), 'pad':0}),
       (layers.MaxPool2DLayer, {'pool_size': (4, 4)}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify, 'W': init.GlorotUniform(gain='relu')}),
       (layers.DropoutLayer, {'p':0.5}),
       (layers.DenseLayer, {'num_units': 512, 'nonlinearity':nonlinearities.leaky_rectify, 'W': init.GlorotUniform(gain='relu')}),
       (layers.DenseLayer, {'num_units': 18, 'nonlinearity':nonlinearities.softmax}),
   ] 

    net = NeuralNet(
        layers=L,
        update=adagrad,
        update_learning_rate=0.01,
        use_label_encoder=True,
        verbose=1,
        max_epochs=100,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        on_epoch_finished=[EarlyStopping(patience=50, criterion='valid_loss')]
        )
    return net

class Classifier(BaseEstimator):

    def __init__(self):
        self.crop_value = 0
        self.gaussianFilter = np.tile(makeGaussian(64-2*self.crop_value, 64-2*self.crop_value-10), (3,1,1)).transpose(1,2,0)
        self.net = build_model(self.crop_value)
        
    def data_augmentation(self, X, y):
        X = sample_from_rotation_x(X)
        y = sample_from_rotation_y(y)
        return X, y

    def preprocess(self, X, transpose=True):
        X = (X / 255.)
        #X = X[:, self.crop_value:64-self.crop_value, self.crop_value:64-self.crop_value, :]
        X = apply_filter(X, self.gaussianFilter)
        X = X.astype(np.float32)
        if transpose:
            X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X, y = self.preprocess(X, False), self.preprocess_y(y)
        X, y = self.data_augmentation(X, y)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)


