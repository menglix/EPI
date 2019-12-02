#####################################
# This implementation is under python 3.5
# Author: Mengli Xiao
#####################################
import numpy as np
import pickle
import h5py
import sys
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score, roc_auc_score, roc_curve
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, AveragePooling2D, Concatenate, Lambda
from keras.layers.merge import add
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
from keras.utils import Sequence

K.set_image_dim_ordering('tf')
from tensorflow import set_random_seed
import random
import argparse


param_ind = 150
file_path = '/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/'
cell_line = 'K562'

feature_maps1ls = [64,128,256]
kernel_size1ls = [3,7,16]
layersls = [2,3,4]
max_poolls = [2,3]
n_FCls = [256,512,800]
n_lyr=layersls[param_ind%3]
feature_map = feature_maps1ls[(param_ind//3)%3]
filterSize = kernel_size1ls[(param_ind//9)%3]
max_poolSize = max_poolls[(param_ind//27)%2]
n_FC = n_FCls[(param_ind//54)%3]

param_ind2 = 249
batchSizels = [32,64,128,256] 
learningRatels = [1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3]
DropoutRatels1 = [0.0,0.2,0.3]
DropoutRatels2 = [0.3,0.5,0.6]
batchSize = batchSizels[param_ind2%4] 
learningRate=learningRatels[(param_ind2//4)%7]
DropoutRate1 = DropoutRatels1[(param_ind2//28)%3]
DropoutRate2 = DropoutRatels2[(param_ind2//84)%3]
# Batch Size: 64|Learning Rate: 0.001|DropoutRate1: 0.3|DropoutRate2: 0.6
X_enhancers = np.load(file_path+cell_line+'/K562enhancer_50_10.npy')
X_promoters = np.load(file_path+cell_line+'/K562promoter_50_10.npy')
labels = np.load(file_path+cell_line+'/K562_labels.npy')
X_enhancers=np_utils.normalize(X_enhancers,axis=0)
X_promoters=np_utils.normalize(X_promoters,axis=0)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def relu(x):
    return Activation('relu')(x)


def evaluate(model,X_enhancers_val,X_promoters_val, test_labels):
    test_pred=model.predict([X_enhancers_val,X_promoters_val],batch_size=60, verbose=2)
    y_pred=(np.round(test_pred)).flatten()
    f1=precision_recall_fscore_support(test_labels, y_pred,average='binary')[2]
    prauc=average_precision_score(test_labels, test_pred)
    ROCauc=roc_auc_score(test_labels,test_pred)
    print('f1 = {:0.4f}. P-R AUC = {:0.4f}. ROC AUC={:0.4f}.'.format(f1,prauc,ROCauc))
    return f1, prauc, ROCauc


class BalancedSequence(Sequence):
    def __init__(self, X1,X2, y, batch_size, process_fn=None):
        """A `Sequence` implementation that returns balanced `y` by undersampling majority class.

        Args:
            X: The numpy array of inputs.
            y: The numpy array of targets.
            batch_size: The generator mini-batch size.
            process_fn: The preprocessing function to apply on `X`
        """
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.batch_size = batch_size
        self.process_fn = process_fn or (lambda x: x)
        self.pos_indices = np.where(y == 1)[0]
        self.neg_indices = np.where(y == 0)[0]
        self.n = min(len(self.pos_indices), len(self.neg_indices))
        self._index_array = None
    def __len__(self):
        # Reset batch after we are done with minority class.
        return (self.n * 2) // self.batch_size
    def on_epoch_end(self):
        # Reset batch after all minority indices are covered.
        self._index_array = None
    def __getitem__(self, batch_idx):
        if self._index_array is None:
            pos_indices = self.pos_indices.copy()
            neg_indices = self.neg_indices.copy()
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)
            self._index_array = np.concatenate((pos_indices[:self.n], neg_indices[:self.n]))
            np.random.shuffle(self._index_array)
        indices = self._index_array[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        return [self.process_fn(self.X1[indices]), self.process_fn(self.X2[indices])], [self.y[indices]]


def fit_model(model):
    model.fit_generator(generator=train_generator, steps_per_epoch=200, 
    epochs=15, verbose=2,
    validation_data = ([X_enhancers_val,X_promoters_val],[labels_val]),callbacks=[EarlyStop])


def neck1(nip, nop, stride):
    def unit(x):
        ident = x
        # in this case, the default stride will be 1
        x = Conv2D(nop,(3,1),
        padding='same',kernel_regularizer=regularizers.l2(1e-5),kernel_initializer='glorot_normal')(x)

        x = BatchNormalization(axis=-1)(x)
        x = relu(x)
        x = Conv2D(nop,(3,1),padding='same',kernel_regularizer=regularizers.l2(1e-5),kernel_initializer='glorot_normal')(x)

        out = add([ident,x])
        return out
    return unit


def neck(nip,nop,stride):
    def unit(x):

        if nip==nop:
            ident = x

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(3,1),
            strides=(stride,stride),padding='same',kernel_regularizer=regularizers.l2(1e-5),kernel_initializer='he_normal')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(3,1),padding='same',kernel_regularizer=regularizers.l2(1e-5),kernel_initializer='he_normal')(x)

            out = add([ident,x])
        else:
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            ident = x
            nbp = nop
            x = Conv2D(nbp,(3,1),
            strides=(stride,stride),padding='same',kernel_regularizer=regularizers.l2(1e-5),kernel_initializer='he_normal')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nbp,(3,1),padding='same',kernel_regularizer=regularizers.l2(1e-5),kernel_initializer='he_normal')(x)


            ident = Conv2D(nop,(1,1),
            strides=(stride,stride),kernel_regularizer=regularizers.l2(1e-5),kernel_initializer='he_normal')(ident)

            out = add([ident,x])

        return out
    return unit

def cake(nip,nop,layers,std):
    def unit(x):
        for i in range(layers):
            if i==0:
                x = neck(nip,nop,std)(x)
            else:
                x = neck(nop,nop,1)(x)
        return x
    return unit

def resrep(nip,nop,blocks,std):
    def unit(x):
        for j in list(range(blocks)):
            if j==0:
                x=neck1(nip,nop,1)(x)
                x=cake(nop,nop,1,1)(x)
                x=MaxPooling2D(pool_size=(2,1))(x)
            else:
                x=cake(nop,nop,2,1)(x)
                x=MaxPooling2D(pool_size=(2,1))(x)
        return x
    return unit
   


def model_3lyr_build_res(feature_maps1,res_layer,kernel_size1, max_pool1,hidden_units):
    opt2=Adam(1e-6)
    input_enhancers = Input(shape=(296,1,22))  # adapt this if using `channels_first` image data format
    i_enhancer = Conv2D(feature_maps1,(kernel_size1,1),strides=(2,1),kernel_regularizer=regularizers.l2(1e-5))(input_enhancers)
    i_enhancer = BatchNormalization(axis=-1)(i_enhancer)
    i_enhancer = relu(i_enhancer)
    i_enhancer = MaxPooling2D(pool_size=(max_pool1,1))(i_enhancer)
    i_enhancer = resrep(feature_maps1,feature_maps1,res_layer,1)(i_enhancer)
    i_enhancer = BatchNormalization(axis=-1)(i_enhancer)
    i_enhancer = relu(i_enhancer)
    i_enhancer = AveragePooling2D(pool_size=(2,1))(i_enhancer)
    input_promoters = Input(shape=(196,1,22)) 
    i_promoter = Conv2D(feature_maps1,(kernel_size1,1),strides=(1,1),kernel_regularizer=regularizers.l2(1e-5))(input_promoters)
    i_promoter = BatchNormalization(axis=-1)(i_promoter)
    i_promoter = relu(i_promoter)
    i_promoter = MaxPooling2D(pool_size=(max_pool1,1))(i_promoter)
    i_promoter = resrep(feature_maps1,feature_maps1,res_layer,1)(i_promoter)
    i_promoter = BatchNormalization(axis=-1)(i_promoter)
    i_promoter = relu(i_promoter)
    i_promoter = AveragePooling2D(pool_size=(2,1))(i_promoter)
    branches = [i_enhancer, i_promoter]
    mixed = Concatenate(axis=1)(branches)
    mixed=Dropout(DropoutRate1)(mixed)
    mixed = BatchNormalization(axis=-1)(mixed)
    i = Flatten()(mixed)
    i = Dense(hidden_units,kernel_regularizer=regularizers.l2(1e-7))(i)
    i = BatchNormalization(axis=-1)(i)
    i = relu(i)
    i=Dropout(DropoutRate2)(i)
    i = Dense(1, activation='sigmoid')(i)
    # create autoencoder model
    model = Model([input_enhancers,input_promoters], i)
    model.compile(loss='binary_crossentropy',optimizer=opt2,metrics=[f1])
    return model


# def evaluate_tf(model,X_enhancers,X_promoters, labels):
    # X_enhancers=np_utils.normalize(X_enhancers,axis=0)
    # X_promoters=np_utils.normalize(X_promoters,axis=0)
    # evaluate(model,(X_enhancers,X_promoters),labels)

def evaluate_model(model):
    print('The validation performance is:')
    a1, a2, a3 = evaluate(model,X_enhancers_val,X_promoters_val,labels_val)
    print('The training performance is:')
    b1, b2, b3 = evaluate(model,X_enhancers_tra1,X_promoters_tra1,labels_tra1)
    print('The test performance is:')
    c1, c2, c3 = evaluate(model,X_enhancers_ts,X_promoters_ts,labels_ts)
    return a1, a2, a3, b1, b2, b3, c1, c2, c3

def retrain():
    K.clear_session()
    set_random_seed(11)
    random.seed(11)
    tf.set_random_seed(11)
    sess = tf.Session(graph=tf.get_default_graph())
    # start a new session
    K.set_session(sess)



#### match key correction 
## previous pandas data 
cell_line = 'K562'
### new_old.npy is the 1-1 correspondance between original targetfinder dataset and the dataset after sorting by label and chromosomes
a = np.load(file_path+cell_line+'/new_old.npy')
X_enhancers = X_enhancers[a]
X_promoters = X_promoters[a]

chromls = np.load(file_path+cell_line+'/chromls.npy').tolist()
a_cum = np.load(file_path+cell_line+'/a_cum.npy').tolist() # positive sample indicator
b_cum = np.load(file_path+cell_line+'/b_cum.npy').tolist() # negative sample indicator
chromls = [x.decode('utf-8') for x in chromls]
chromls2 = list(chromls[:-3])+[chromls[-1]]

# chrom = chromls2[ind]
chrom = 'chr1'
i = chromls.index(chrom)
np.random.seed(9)
if i == 0:
    ts_ind = list(range(a_cum[i]))+list(range(a_cum[-1],b_cum[i]))
else:
    ts_ind = list(range(a_cum[i-1],a_cum[i]))+list(range(a_cum[-1]+b_cum[i-1],a_cum[-1]+b_cum[i]))
np.random.shuffle(ts_ind)
X_enhancers_ts = X_enhancers[ts_ind]
X_promoters_ts = X_promoters[ts_ind]
labels_ts = labels[ts_ind]

np.random.seed(9)
val_ind = list(range(a_cum[19],a_cum[21]))+list(range(a_cum[-1]+b_cum[19],a_cum[-1]+b_cum[21]))
np.random.shuffle(val_ind)
X_enhancers_val = X_enhancers[val_ind]
X_promoters_val = X_promoters[val_ind]
labels_val = labels[val_ind]


np.random.seed(9)
tra_ind = [x for x in list(range(X_enhancers.shape[0])) if x not in ts_ind+val_ind]
np.random.shuffle(tra_ind)
X_enhancers_tra1 = X_enhancers[tra_ind]
X_promoters_tra1 = X_promoters[tra_ind]
labels_tra1 = labels[tra_ind]

train_generator=BalancedSequence(X_enhancers_tra1,X_promoters_tra1,labels_tra1,batch_size=batchSize)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels_tra1),
                                                 labels_tra1)
class_weight_dict = dict(enumerate(class_weights))


set_random_seed(11)

EarlyStop=EarlyStopping(monitor='val_f1',mode='max',patience=10)
path='/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/K562/Models/Genomic1/'+'best_'+chrom+'_resCNN.hdf5'
checkpoint = ModelCheckpoint(path, 
                    monitor='val_f1', verbose=0, mode='max',save_best_only=True)

# retrain()
model = model_3lyr_build_res(feature_map,n_lyr,filterSize,max_poolSize,n_FC)
model.summary()
print('the current test chromosome is '+chrom)
fit_model(model)
# model.load_weights(path)
results['chrom'].append(chrom)
evaluate_model(model)
