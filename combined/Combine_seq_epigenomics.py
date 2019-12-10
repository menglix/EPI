#####################################
# This implementation is under python 3.5
# Author: Mengli Xiao
#####################################
import numpy as np
import pickle
import h5py
import sys
import scipy
from keras.layers import Activation, multiply, Dropout, Flatten, Dense, Input, Conv2D, Convolution1D, MaxPooling1D, MaxPooling2D, AveragePooling2D, Concatenate, Lambda
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Model
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import add
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold,train_test_split, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow import set_random_seed
import random

enhancer_length = 600 # TODO: get this from input
promoter_length = 400 # TODO: get this from input
n_kernels = 256 # Number of kernels; used to be 1024
filter_length = 8 # Length of each kernel
#LSTM_out_dim = 50 # Output direction of ONE DIRECTION of LSTM; used to be 512
dense_layer_size = 800
num_epochs = 12
num_epochs_pre = 10
learningRate = 1e-5
batch_size = 64
weightDecay = 0.0
file_path = '/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/'
cell_line = 'K562'
batchSize = 200

### we used f1 for model training metrics
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

def relu(x):
    return Activation('relu')(x)


def evaluate(model, X_enhancers_val,X_promoters_val, test_labels):
    test_pred=model.predict([X_enhancers_val,X_promoters_val],batch_size=60, verbose=2)
    y_pred=(np.round(test_pred)).flatten()
    f1=precision_recall_fscore_support(test_labels, y_pred,average='binary')[2]
    prauc=average_precision_score(test_labels, test_pred)
    ROCauc=roc_auc_score(test_labels,test_pred)
    print('f1 = {:0.4f}. P-R AUC = {:0.4f}. ROC AUC={:0.4f}.'.format(f1,prauc,ROCauc))
    return f1, prauc, ROCauc

def evaluate_model(model):
    print('The validation performance is:')
    a1, a2, a3 = evaluate(model,X_enhancers_val,X_promoters_val,labels_val)
    print('The training performance is:')
    b1, b2, b3 = evaluate(model,X_enhancers_tra1,X_promoters_tra1,labels_tra1)
    print('The test performance is:')
    c1, c2, c3 = evaluate(model,X_enhancers_ts,X_promoters_ts,labels_ts)
    return a1, a2, a3, b1, b2, b3, c1, c2, c3

### epigenomic model structure
### we expanded the dimension of data from (296,22) to (296,1,22) to use the Conv2d function, but the it could be data with shape (296,22) and use Conv1D
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

## sequence model structure
def seq_attentionmodel(enhancer_length = 600,promoter_length = 400,n_kernels = 256,
                       filter_length = 8,dense_layer_size = 800):
    opt = Adam(lr = 1e-5)
    input_enh_sequence = Input(shape=(enhancer_length,4))
    enh_seq = Convolution1D(input_dim = 4,
                                            input_length = enhancer_length,
                                            nb_filter = n_kernels,
                                            filter_length = filter_length,
                                            border_mode = "valid",
                                            subsample_length = 1,
                                            kernel_regularizer = l2(2e-5))(input_enh_sequence)
    enh_seq = relu(enh_seq)
    enh_seq = MaxPooling1D(pool_length = int(filter_length/2), stride = int(filter_length/2))(enh_seq)
    input_prom_sequence = Input(shape=(promoter_length,4))
    prom_seq = Convolution1D(input_dim = 4,
                                            input_length = promoter_length,
                                            nb_filter = n_kernels,
                                            filter_length = filter_length,
                                            border_mode = "valid",
                                            subsample_length = 1,
                                            kernel_regularizer = l2(2e-5))(input_prom_sequence)
    #prom_seq = BatchNormalization(momentum=0.997)(prom_seq)
    prom_seq = relu(prom_seq)
    prom_seq = MaxPooling1D(pool_length = int(filter_length/2), stride = int(filter_length/2))(prom_seq)
    seq_mixed = Concatenate(axis=1)([enh_seq, prom_seq])
    seq_mixed = BatchNormalization(momentum=0.997)(seq_mixed)
    seq_mixed = Dropout(0.2)(seq_mixed)
    seq_mixed = Flatten()(seq_mixed)
    seq_mixed = Dense(output_dim=dense_layer_size,
                        init="glorot_uniform",
                        activity_regularizer=l2(1e-6))(seq_mixed)
    seq_mixed = BatchNormalization(momentum=0.997)(seq_mixed)
    seq_mixed = relu(seq_mixed)
    a_prob = Dense(dense_layer_size, activation='softmax', name='attention_vec',kernel_regularizer=l2(weightDecay))(seq_mixed)
    attention_mul = multiply([seq_mixed, a_prob], name='attention_mul')
    seq_mixed = Dense(256,kernel_regularizer=l2(weightDecay))(attention_mul)
    a_prob = Dense(256, activation='softmax', name='attention_vec1',kernel_regularizer=l2(weightDecay/10))(seq_mixed)
    attention_mul = multiply([seq_mixed, a_prob], name='attention_mul1')
    attention_mul = Dense(128,kernel_regularizer=l2(weightDecay/10))(attention_mul)
    attention_mul = BatchNormalization(momentum=0.997)(attention_mul)
    attention_mul = relu(attention_mul)
 #seq_mixed = Dropout(0.5)(seq_mixed)
    seq_mixed = Dense(1)(attention_mul)
    seq_mixed = BatchNormalization(momentum=0.997, scale=False)(seq_mixed)
    seq_mixed = Activation('sigmoid')(seq_mixed)
    model = Model([input_enh_sequence,input_prom_sequence], seq_mixed)
    model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics = [f1])
    model.summary()
    return model

def convertCNNfeature(data):
    cnnFeatures = intermediate_lyr_model1.predict(data,batch_size=100, verbose=2)
    npCNNFeatures = np.array(cnnFeatures)
    return npCNNFeatures


#### match key correction 
## previous pandas data 
cell_line = 'K562'
X_enhancers = np.load(file_path+cell_line+'/K562enhancer_50_10.npy')
X_promoters = np.load(file_path+cell_line+'/K562promoter_50_10.npy')
labels = np.load(file_path+cell_line+'/K562_labels.npy')
X_enhancers=np_utils.normalize(X_enhancers,axis=0)
X_promoters=np_utils.normalize(X_promoters,axis=0)
cell_line = 'K562'
### new_old.npy is the 1-1 correspondance between original targetfinder dataset and the dataset after sorting by label and chromosomes
## The .npy files is generated from pickle file in python 2.7 script, so we need to use .npy as a file connector
## The following steps could be further optimized if the model training and data preprocessing is done within the same python version
a = np.load(file_path+cell_line+'/new_old.npy')
X_enhancers = X_enhancers[a]
X_promoters = X_promoters[a]
## genomic data comes from this pandas dataframe, but this dataframe was re-ordered before matching with sequence
# this is used to accommodate lacking of the package in the Python 3.5 in the GPU cluster computing platform, could be simplified if keras and the package pandas are installed at the same time.
chromls = np.load(file_path+cell_line+'/chromls.npy').tolist()
a_cum = np.load(file_path+cell_line+'/a_cum.npy').tolist() # positive sample indicator for all chromosomes
b_cum = np.load(file_path+cell_line+'/b_cum.npy').tolist() # negative sample indicator for all chromosomes
chromls = [x.decode('utf-8') for x in chromls]
chromls2 = list(chromls[:-3])+[chromls[-1]]


#### if the test chromosome is chr1, ind=0, o.w., the ind could be any integer number within the interval [0,21]
ind = 0 
chrom = chromls2[ind]
#for i, chrom in enumerate(chromls[0:11]):
i=chromls.index(chrom)
ts_ind = list(range(a_cum[i-1]*(i!=0),a_cum[i]))+list(range(a_cum[-1]+b_cum[i-1]*(i!=0),a_cum[-1]+b_cum[i]))
##### chromls[1:11] is for using chromosome 10 to chromosome 19 as the test chromosome
## ensure we get the same chromsome split at every run
np.random.seed(9)
# use chr1 as test, chr 8 and chr9 as validation, and the rest as training
## ensure we get the same chromsome split at every run
## a_cum[0] is the start index of the chr1, I split chr1 as the test data
np.random.shuffle(ts_ind)
np.random.seed(9)
val_ind = list(range(a_cum[19],a_cum[21]))+list(range(a_cum[-1]+b_cum[19],a_cum[-1]+b_cum[21]))
np.random.shuffle(val_ind)
np.random.seed(9)
tra_ind = [x for x in list(range(X_enhancers.shape[0])) if x not in ts_ind+val_ind]
tra_ind = [x for x in list(range(41477)) if x not in ts_ind+val_ind]

np.random.shuffle(tra_ind)
#### The 20776 observation in epigenomics corresponds to the last observation in sequence, so removed
#### The epigenomic data also needs to delete this observation
for d_id, indls in enumerate([tra_ind,val_ind,ts_ind]):
    if 20776 in indls:
        del indls[indls.index(20776)]
    tra_ind = [x-1 if x>20776 else x for x in tra_ind]
    val_ind = [x-1 if x>20776 else x for x in val_ind]
    ts_ind = [x-1 if x>20776 else x for x in ts_ind]

np.delete(X_enhancers,20776,0)
np.delete(X_promoters,20776,0)


X_enhancers_ts = X_enhancers[ts_ind]
X_promoters_ts = X_promoters[ts_ind]
labels_ts = labels[ts_ind]

## a_cum[19], a_cum[21] is the start index of the chr8 and end index of chr[9] separately, we split them as our val data

X_enhancers_val = X_enhancers[val_ind]
X_promoters_val = X_promoters[val_ind]
labels_val = labels[val_ind]

#num_tra = len(seq_order)-len(ts_ind)-len(val_ind)

X_enhancers_tra1 = X_enhancers[tra_ind]
X_promoters_tra1 = X_promoters[tra_ind]
labels_tra1 = labels[tra_ind]


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels_tra1),
                                                 labels_tra1)
class_weight_dict = dict(enumerate(class_weights))


set_random_seed(11)

EarlyStop=EarlyStopping(monitor='val_f1',mode='max',patience=10)
path='/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/K562/Models/Genomic1/'+'Original_'+chrom+'_bBach_CNN_50_10v1_1_2.hdf5'
checkpoint = ModelCheckpoint(path, 
                    monitor='val_f1', verbose=0, mode='max',save_best_only=True)

# retrain()
model1 = model_3lyr_build2(256,256,256,16,16,16,2,2,2,512)
model1.load_weights(path)
valf1, valpr,valroc, trainf1, trainpr, trainroc, tsf1, tspr, tsroc = evaluate_model(model1)

### extract the features from teh layer from the 512 fully-connected layer in the right-hand-side box (red) in Figure 3
intermediate_lyr_model1 = Model(inputs=model1.input,outputs=model1.get_layer('activation_7').output)

CNN2_tra=convertCNNfeature([X_enhancers_tra1,X_promoters_tra1])
CNN2_val = convertCNNfeature([X_enhancers_val,X_promoters_val])
CNN2_ts = convertCNNfeature([X_enhancers_ts,X_promoters_ts])
del X_enhancers, X_promoters, X_enhancers_tra1,X_promoters_tra1

#==============================================================================
# sequence model
#==============================================================================

file_path = '/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/K562/'
S_enhancers = np.load(file_path+'K562_enhancer.npy')
S_promoters = np.load(file_path+'K562_promoter.npy')
labels = np.load(file_path+'K562_labels.npy')


S_enhancers = np.swapaxes(S_enhancers,1,2)
S_promoters = np.swapaxes(S_promoters,1,2)


## re-order the sequence data to follow the epigenomics order
## The .npy files is generated from pickle file in python 2.7 script, so we need to use .npy as a file connector
## The following steps could be further optimized if the model training and data preprocessing is done within the same python version
file_path = '/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/'
seq_order = np.load(file_path+'Seq_order.npy') # re-order sequence with the same order as epigenomic
S_enhancers = S_enhancers[seq_order]
S_promoters = S_promoters[seq_order]
labels = labels[seq_order]

#==============================================================================
# now the sequence has the same order as genomics
#==============================================================================
S_enhancers_ts = S_enhancers[ts_ind]
S_promoters_ts = S_promoters[ts_ind]
labels_ts = labels[ts_ind]
S_enhancers_val = S_enhancers[val_ind]
S_promoters_val = S_promoters[val_ind]
labels_val = labels[val_ind]
S_enhancers_tra = S_enhancers[tra_ind]
S_promoters_tra = S_promoters[tra_ind]
labels_tra = labels[tra_ind]

del S_enhancers, S_promoters
### took the central region of the sequence data
S_enhancers_tra = S_enhancers_tra[:, 1200:1800, :]
S_promoters_tra = S_promoters_tra[:, 800:1200, :]
S_enhancers_ts = S_enhancers_ts[:, 1200:1800, :]
S_promoters_ts = S_promoters_ts[:, 800:1200, :]
S_enhancers_val = S_enhancers_val[:, 1200:1800, :]
S_promoters_val = S_promoters_val[:, 800:1200, :]


##### sequence model path
path=file_path+cell_line+'/Models/Seq/'+chrom+'_'+str(ind)+'_CNN_attention_sigmoid.hdf5'
      
model2 = seq_attentionmodel()
model2.load_weights(path)
print('the current test chromosome is '+chrom)
evaluate(model2,S_enhancers_ts,S_promoters_ts,labels_ts)
# 128 fc in the sequence model in Figure 3
intermediate_lyr_model1 = Model(inputs=model2.input,outputs=model2.get_layer('activation_11').output)
SeqCNN_tra=convertCNNfeature([S_enhancers_tra,S_promoters_tra])
SeqCNN_val = convertCNNfeature([S_enhancers_val,S_promoters_val])
SeqCNN_ts = convertCNNfeature([S_enhancers_ts,S_promoters_ts])


del S_enhancers_tra,S_promoters_tra, S_enhancers_val,S_promoters_val,S_enhancers_ts,S_promoters_ts

#### combine the epigenomic and sequence feature as a single input to the combined neural network model 
tra = np.concatenate((SeqCNN_tra,CNN2_tra),axis=1)
val = np.concatenate((SeqCNN_val,CNN2_val),axis=1)
ts = np.concatenate((SeqCNN_ts,CNN2_ts),axis=1)

#### standardize the input
scaler1 = StandardScaler()
scaler1.fit(tra)
tra = scaler1.transform(tra)
scaler2 = StandardScaler()
scaler2.fit(val)
val = scaler1.transform(val)
scaler3 = StandardScaler()
scaler3.fit(ts)
ts = scaler1.transform(ts)

np.random.seed(9)
param_ind=116 ## selected combined model parameters
dense1ls = [64,128,256,512,1024]
learningls=[1e-4,1e-5,1e-6]

## combined model structure
## structure is as shown in Figure 3
dense1 = dense1ls[param_ind%5] ## 1st layer in the combined model is 128
dense2 = dense1ls[(param_ind//5)%5] ## 2nd layer in the combined model is 512
learningRate = learningls[(param_ind//25)%3] ## 1e-5
lyrnum=[1,2]
n_lyr = lyrnum[(param_ind//75)%2] ## layer is 2


def combine_model():
    opt2 = Adam(learningRate)
    input_feature = Input(shape=(tra.shape[1],))
    if n_lyr == 2:
        seq_mixed = Dense(output_dim=dense1,init="glorot_normal",kernel_regularizer=l2(1e-6))(input_feature)
        seq_mixed = BatchNormalization(axis=-1)(seq_mixed)
        seq_mixed = relu(seq_mixed)
        seq_mixed = Dropout(0.3,seed=9)(seq_mixed)
        seq_mixed = Dense(output_dim=dense2,init="glorot_normal",kernel_regularizer=l2(1e-6))(seq_mixed)
        seq_mixed = BatchNormalization(axis=-1)(seq_mixed)
        seq_mixed = relu(seq_mixed)
        seq_mixed = Dropout(0.5,seed=9)(seq_mixed)
    else:
        seq_mixed = Dense(output_dim=dense1,init="glorot_normal",kernel_regularizer=l2(1e-6))(input_feature)
        seq_mixed = BatchNormalization(axis=-1)(seq_mixed)
        seq_mixed = relu(seq_mixed)
        seq_mixed = Dropout(0.3,seed=9)(seq_mixed)
    seq_mixed = Dense(1,activation='sigmoid')(seq_mixed)
    model = Model(input_feature, seq_mixed)
    model.compile(loss = 'binary_crossentropy',optimizer = opt2,metrics = [f1])
    return model

model_c = combine_model()
model_c.summary()
EarlyStop=EarlyStopping(monitor='val_f1',mode='max',patience=10)
path = '/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/K562/Models/Combine/'+chrom+str(param_ind)+'_v5.hdf5'
checkpoint = ModelCheckpoint(path, 
                    monitor='val_f1', verbose=2, mode='max',save_best_only=True)
model_c.fit(tra,
          [labels_tra],
          validation_data = (val, labels_val),
          batch_size = 200,
          nb_epoch = 300,verbose=2,
          shuffle = True,
          class_weight=class_weight_dict,
          callbacks=[EarlyStop,checkpoint]#checkpointer]
          )

model_c.load_weights(path)


def evaluate(model, test, test_labels):
    test_pred=model.predict(test,batch_size=60, verbose=2)
    y_pred=(np.round(test_pred)).flatten()
    f1=precision_recall_fscore_support(test_labels, y_pred,average='binary')[2]
    prauc=average_precision_score(test_labels, test_pred)
    ROCauc=roc_auc_score(test_labels,test_pred)
    print('f1 = {:0.4f}. P-R AUC = {:0.4f}. ROC AUC={:0.4f}.'.format(f1,prauc,ROCauc))
    return f1, prauc, ROCauc

evaluate(model_c,val,labels_val)
evaluate(model_c,tra,labels_tra)
evaluate(model_c,ts,labels_ts)

