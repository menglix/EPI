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
K.set_image_dim_ordering('tf')
from tensorflow import set_random_seed
import random


file_path = '/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/'
cell_line = 'K562'
X_enhancers = np.load(file_path+cell_line+'/K562enhancer_50_10.npy')
X_promoters = np.load(file_path+cell_line+'/K562promoter_50_10.npy')
labels = np.load(file_path+cell_line+'/K562_labels.npy')
X_enhancers=np_utils.normalize(X_enhancers,axis=-1)
X_promoters=np_utils.normalize(X_promoters,axis=-1)


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

'''
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred, curve='ROC')[1]
    K.get_session().rn(tf.local_variables_initializer())
    return auc
'''
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def relu(x):
    return Activation('relu')(x)


def evaluate(model,X_enhancers_val,X_promoters_val, test_labels):
    test_pred=model.predict([X_enhancers_val,X_promoters_val],batch_size=60, verbose=1)
    y_pred=(np.round(test_pred)).flatten()
    f1=precision_recall_fscore_support(test_labels, y_pred,average='binary')[2]
    prauc=average_precision_score(test_labels, test_pred)
    ROCauc=roc_auc_score(test_labels,test_pred)
    print('f1 = {:0.4f}. P-R AUC = {:0.4f}. ROC AUC={:0.4f}.'.format(f1,prauc,ROCauc))
    return f1, prauc, ROCauc


def fit_initial_model(model):
    model.fit([X_enhancers_tra, X_promoters_tra],
                      [labels_aug_tra],
                      batch_size = 200,
                      epochs = 20,
                      shuffle = True,validation_data=([X_enhancers_val,X_promoters_val],labels_val), callbacks=[EarlyStop,checkpoint])


def fit_model(model):
    model.fit([X_enhancers_tra1, X_promoters_tra1],
                      [labels_tra1],
                      batch_size = 200,
                      epochs = 300,verbose=2,
                      shuffle = True,validation_data=([X_enhancers_val,X_promoters_val],labels_val), class_weight=class_weight_dict,callbacks=[EarlyStop,checkpoint])


def model_3lyr_build2(feature_maps1,feature_maps2,feature_maps3,kernel_size1,kernel_size2,kernel_size3, max_pool1,max_pool2,max_pool3,hidden_units):
    opt2=Adam(1e-6)
    input_enhancers = Input(shape=(296,1,22))  # adapt this if using `channels_first` image data format
    i_enhancer = Conv2D(feature_maps1,(kernel_size1,1),strides=(1,1),kernel_regularizer=regularizers.l2(1e-5))(input_enhancers)
    i_enhancer = BatchNormalization(axis=-1)(i_enhancer)
    i_enhancer = relu(i_enhancer)
    i_enhancer = MaxPooling2D(pool_size=(max_pool1,1))(i_enhancer)
    i_enhancer = Conv2D(feature_maps2,(kernel_size2,1),strides=(1,1),kernel_regularizer=regularizers.l2(1e-5))(i_enhancer)
    i_enhancer = BatchNormalization(axis=-1)(i_enhancer)
    i_enhancer = relu(i_enhancer)
    i_enhancer = MaxPooling2D(pool_size=(max_pool2,1))(i_enhancer)
    i_enhancer = Conv2D(feature_maps3,(kernel_size3,1),strides=(1,1),kernel_regularizer=regularizers.l2(1e-5))(i_enhancer)
    i_enhancer = BatchNormalization(axis=-1)(i_enhancer)
    i_enhancer = relu(i_enhancer)
    i_enhancer=Dropout(0.2)(i_enhancer)
    i_enhancer = MaxPooling2D(pool_size=(max_pool3,1))(i_enhancer)
    input_promoters = Input(shape=(196,1,22)) 
    i_promoter = Conv2D(feature_maps1,(kernel_size1,1),strides=(1,1),kernel_regularizer=regularizers.l2(1e-5))(input_promoters)
    i_promoter = BatchNormalization(axis=-1)(i_promoter)
    i_promoter = relu(i_promoter)
    i_promoter = MaxPooling2D(pool_size=(max_pool1,1))(i_promoter)
    i_promoter = Conv2D(feature_maps2,(kernel_size2,1),strides=(1,1),kernel_regularizer=regularizers.l2(1e-5))(i_promoter)
    i_promoter = BatchNormalization(axis=-1)(i_promoter)
    i_promoter = relu(i_promoter)
    i_promoter = MaxPooling2D(pool_size=(max_pool2,1))(i_promoter)
    i_promoter = Conv2D(feature_maps3,(kernel_size3,1),strides=(1,1),kernel_regularizer=regularizers.l2(1e-5))(i_promoter)
    i_promoter = BatchNormalization(axis=-1)(i_promoter)
    i_promoter = relu(i_promoter)
    i_promoter=Dropout(0.2)(i_promoter)
    i_promoter = MaxPooling2D(pool_size=(max_pool3,1))(i_promoter)
    branches = [i_enhancer, i_promoter]
    mixed = Concatenate(axis=1)(branches)
    mixed=Dropout(0.3)(mixed)
    mixed = BatchNormalization(axis=-1)(mixed)
    i = Flatten()(mixed)
    i = Dense(hidden_units,kernel_regularizer=regularizers.l2(1e-7))(i)
    i = BatchNormalization(axis=-1)(i)
    i = relu(i)
    i=Dropout(0.6)(i)
    i = Dense(1, activation='sigmoid')(i)
    # create autoencoder model
    model = Model([input_enhancers,input_promoters], i)
    model.compile(loss='binary_crossentropy',optimizer=opt2,metrics=[f1])
    return model

def evaluate_tf(model,X_enhancers,X_promoters, labels):
    X_enhancers=np_utils.normalize(X_enhancers,axis=0)
    X_promoters=np_utils.normalize(X_promoters,axis=0)
    evaluate(model,(X_enhancers,X_promoters),labels)

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
### see otherFiles.py for how the .npy files below were generated
### new_old.npy is the 1-1 correspondance between original targetfinder dataset and the dataset after sorting by label and chromosomes
## The .npy files is generated from pickle file in python 2.7 script, so we need to use .npy as a file connector
## The following steps could be further optimized if the model training and data preprocessing is done within the same python version
a = np.load(file_path+cell_line+'/new_old.npy')
X_enhancers = X_enhancers[a]
X_promoters = X_promoters[a]

## genomic data comes from this pandas dataframe, but this dataframe was re-ordered before matching with sequence
# this is used to accommodate lacking of the package in the Python 3.5 in the GPU cluster computing platform, could be simplified if keras and the package pandas are installed at the same time.
chromls = np.load(file_path+cell_line+'/chromls.npy').tolist()
a_cum = np.load(file_path+cell_line+'/a_cum.npy').tolist() # positive sample indicator for each chromosome, it records the cumulative index of samples in each chromosome
b_cum = np.load(file_path+cell_line+'/b_cum.npy').tolist() # negative sample indicator for each chromosome, it records the cumulative index of samples in each chromosome
chromls = [x.decode('utf-8') for x in chromls]
chromls2 = list(chromls[:-3])+[chromls[-1]]



#### chromls2 contains chromosomes other than the validation chromosome
#### The index corresponds to the order in epigenomics data
### chromosome index (ind) ranges from 0 to 21
ind = 0 # if the test chromosome is 1
chrom = chromls2[ind]
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


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels_tra1),
                                                 labels_tra1)
class_weight_dict = dict(enumerate(class_weights))


set_random_seed(11)

EarlyStop=EarlyStopping(monitor='val_f1',mode='max',patience=10)
path='/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/K562/Models/Genomic1/'+'Original_'+chrom+'_CNN_50_10v1_1_2.hdf5'
checkpoint = ModelCheckpoint(path, 
                    monitor='val_f1', verbose=0, mode='max',save_best_only=True)

# retrain()
model = model_3lyr_build2(256,256,256,16,16,16,2,2,2,512)
fit_model(model)
model.load_weights(path)
print('the current test chromosome is '+chrom)
evaluate_model(model)

