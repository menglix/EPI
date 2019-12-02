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
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from sklearn.model_selection import StratifiedKFold,train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import pickle
from keras import backend as K
from keras.utils import to_categorical
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

param_ind = 30
kernel_ls = [64,128,100]
res_layerls=[1,2,3,4]
res_filterls=[3,4,8,10]
n_kernels = kernel_ls[param_ind%3] #64
n_lyr = res_layerls[(param_ind//3)%4] #2
filter_res = res_filterls[(param_ind//12)%4] #8
enhancer_length = 600 # TODO: get this from input
promoter_length = 400 # TODO: get this from input
# n_kernels = 256 # Number of kernels; used to be 1024
filter_length = 4 # Length of each kernel
#LSTM_out_dim = 50 # Output direction of ONE DIRECTION of LSTM; used to be 512
dense_layer_size = 800
num_epochs = 12
num_epochs_pre = 10
learningRate_ls = [1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3]
batchSize_ls = [32,64,128,256]
weightDecay_ls = [0.0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5]
learningRate = 1e-5
batch_size = 200


#kernel_size  = 256


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
    b1, b2, b3 = evaluate(model,X_enhancers_tra,X_promoters_tra,labels_tra)
    print('The test performance is:')
    c1, c2, c3 = evaluate(model,X_enhancers_ts,X_promoters_ts,labels_ts)
    return a1, a2, a3, b1, b2, b3, c1, c2, c3

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def accumulate(lis):
    total = 0
    for item in lis:
        total += item
        yield total

    enh_seq = Convolution1D(input_dim = 4,
                                            input_length = enhancer_length,
                                            nb_filter = n_kernels,
                                            filter_length = filter_length,
                                            border_mode = "valid",
                                            subsample_length = 1,
                                            kernel_regularizer = l2(2e-5))(input_enh_sequence)
def neck1(nip, nop, stride):
    def unit(x):
        ident = x
        # in this case, the default stride will be 1
        x = Convolution1D(nop,filter_res,border_mode='same',W_regularizer=l2(2e-5),kernel_initializer='glorot_normal')(x)
        x = BatchNormalization(axis=-1)(x)
        x = relu(x)
        x = Convolution1D(nop,filter_res,border_mode='same',W_regularizer=l2(2e-5),kernel_initializer='glorot_normal')(x)
        out = add([ident,x])
        return out
    return unit


def neck(nip,nop,stride):
    def unit(x):
        if nip==nop:
            ident = x
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution1D(nop,filter_res,border_mode='same',W_regularizer=l2(2e-5),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution1D(nop,filter_res,border_mode='same',W_regularizer=l2(2e-5),kernel_initializer='glorot_normal')(x)
            out = add([ident,x])
        else:
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            ident = x
            nbp = nop
            x = Convolution1D(nop,filter_res,border_mode='same',W_regularizer=l2(2e-5),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution1D(nop,filter_res,border_mode='same',W_regularizer=l2(2e-5),kernel_initializer='glorot_normal')(x)
            ident = Convolution1D(nop,1,subsample_length=stride,W_regularizer=l2(2e-5),kernel_initializer='glorot_normal')(ident)
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
                x=MaxPooling1D(pool_length=2)(x)
            else:
                x=cake(nop,nop,2,1)(x)
                x=MaxPooling1D(pool_length=2)(x)
        return x
    return unit
   


def seq_model():
    opt = Adam(lr = learningRate)
    input_enh_sequence = Input(shape=(enhancer_length,4))
    enh_seq = Convolution1D(input_dim = 4,
                                            input_length = enhancer_length,
                                            nb_filter = n_kernels,
                                            filter_length = 8,
                                            border_mode = "valid",
                                            subsample_length = 1,
                                            kernel_regularizer = l2(2e-5))(input_enh_sequence)
    enh_seq = relu(enh_seq)
    enh_seq = MaxPooling1D(pool_length = 10, stride = 10)(enh_seq)
    enh_seq = resrep(n_kernels,n_kernels,n_lyr,1)(enh_seq)
    input_prom_sequence = Input(shape=(promoter_length,4))
    prom_seq = Convolution1D(input_dim = 4,
                                            input_length = promoter_length,
                                            nb_filter = n_kernels,
                                            filter_length = 8,
                                            border_mode = "valid",
                                            subsample_length = 1,
                                            kernel_regularizer = l2(2e-5))(input_prom_sequence)
    prom_seq = relu(prom_seq)
    prom_seq = MaxPooling1D(pool_length = 10, stride = 10)(prom_seq)
    prom_seq = resrep(n_kernels,n_kernels,n_lyr,1)(prom_seq)
    seq_mixed = Concatenate(axis=1)([enh_seq, prom_seq])
    seq_mixed = BatchNormalization(momentum=0.997)(seq_mixed)
    seq_mixed = relu(seq_mixed)
    seq_mixed = Dropout(0.5)(seq_mixed)
    seq_mixed = Flatten()(seq_mixed)
    seq_mixed = Dense(1)(seq_mixed)
    seq_mixed = BatchNormalization(momentum=0.997, scale=False)(seq_mixed)
    seq_mixed = Activation('sigmoid')(seq_mixed)
    model = Model([input_enh_sequence,input_prom_sequence], seq_mixed)
    model.compile(loss = 'binary_crossentropy',
                optimizer = opt,
                metrics = [f1])
    model.summary()
    return model

def fit_model(model):
    model.fit([X_enhancers_tra, X_promoters_tra],
          [labels_tra],
          validation_data = ([X_enhancers_val, X_promoters_val], labels_val),
          batch_size = 200,
          nb_epoch = 300,verbose=2,
          shuffle = True,
          class_weight = class_weight_dict,
          callbacks=[EarlyStop, checkpoint])

file_path = '/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/K562/'
X_enhancers = np.load(file_path+'K562_enhancer.npy')
X_promoters = np.load(file_path+'K562_promoter.npy')
labels = np.load(file_path+'K562_labels.npy')



### last sample in sequence not available from SPEID

labels = labels[:41476]
X_enhancers = X_enhancers[:41476]
X_promoters = X_promoters[:41476]

X_enhancers = np.swapaxes(X_enhancers,1,2)
X_promoters = np.swapaxes(X_promoters,1,2)
print('Now the shape is:'+str(X_enhancers.shape))

## match key 
file_path='/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/'
cell_line='K562'

## re-order the sequence data to follow the epigenomics order
seq_order = np.load(file_path+'/Seq_order.npy')
X_enhancers = X_enhancers[seq_order]
X_promoters = X_promoters[seq_order]
labels = labels[seq_order]

chromls = np.load(file_path+cell_line+'/chromls.npy').tolist()
a_cum = np.load(file_path+cell_line+'/a_cum.npy').tolist() # positive sample indicator
b_cum = np.load(file_path+cell_line+'/b_cum.npy').tolist() # negative sample indicator
chromls = [x.decode('utf-8') for x in chromls]
chromls2 = list(chromls[:-3])+[chromls[-1]]


#### chromls2 contains chromosomes other than the validation chromosome
#### The index corresponds to the order in epigenomics data


chrom = chromls2[ind]
i=chromls.index(chrom)
ts_ind = list(range(a_cum[i-1]*(i!=0),a_cum[i]))+list(range(a_cum[-1]+b_cum[i-1]*(i!=0),a_cum[-1]+b_cum[i]))
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

for d_id, indls in enumerate([tra_ind,val_ind,ts_ind]):
    if 20776 in indls:
        del indls[indls.index(20776)]
    tra_ind = [x-1 if x>20776 else x for x in tra_ind]
    val_ind = [x-1 if x>20776 else x for x in val_ind]
    ts_ind = [x-1 if x>20776 else x for x in ts_ind]


X_enhancers_ts = X_enhancers[ts_ind]
X_promoters_ts = X_promoters[ts_ind]
labels_ts = labels[ts_ind]

## a_cum[19], a_cum[21] is the start index of the chr8 and end index of chr[9] separately, we split them as our val data

X_enhancers_val = X_enhancers[val_ind]
X_promoters_val = X_promoters[val_ind]
labels_val = labels[val_ind]

#num_tra = len(seq_order)-len(ts_ind)-len(val_ind)

X_enhancers_tra = X_enhancers[tra_ind]
X_promoters_tra = X_promoters[tra_ind]
labels_tra = labels[tra_ind]


X_enhancers_tra = X_enhancers_tra[:, 1200:1800, :]
X_promoters_tra = X_promoters_tra[:, 800:1200, :]
X_enhancers_ts = X_enhancers_ts[:, 1200:1800, :]
X_promoters_ts = X_promoters_ts[:, 800:1200, :]
X_enhancers_val = X_enhancers_val[:, 1200:1800, :]
X_promoters_val = X_promoters_val[:, 800:1200, :]

############################################ use this to calculate the class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels_tra),
                                                 labels_tra)
class_weight_dict = dict(enumerate(class_weights))
class_weight_dict={0:1., 1:800.}
########################################################################### change this path 
path=file_path+cell_line+'/Models/Seq/'+chrom+'_'+str(param_ind)+'_CNN.hdf5'
EarlyStop=EarlyStopping(monitor='val_f1',mode='max',patience=10)
checkpoint = ModelCheckpoint(path, 
                    monitor='val_f1', verbose=2, mode='max',save_best_only=True)
                    
model = seq_model()
fit_model(model)
model.load_weights(path)

evaluate_model(model)
