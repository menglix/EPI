#####################################
# This implementation is under python 2.7, because the match is conducted in python 2.7, but the implementation platform for deep learning model is python 3.5
# Author: Mengli Xiao
#####################################
import pandas as pd
import pickle
import numpy as np

### similar function to cumsum() in R
def accumulate(lis):
    total = 0
    for item in lis:
        total += item
        yield total


## genomic data comes from this pandas dataframe, but this dataframe was re-ordered before matching with sequence
# the new order has 1. positive pairs first and then the negative pairs
                  # 2. sorted by enhancer chromosome and enhancer start location
file_path='/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/'
cell_line='K562'

cols = ['label','enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'enhancer_name', 'promoter_name','enhancer_distance_to_promoter']
training_df = pd.read_csv(file_path+cell_line+'/training.csv.gz',compression='gzip',engine='python',usecols=cols)

training_df = pd.read_csv(file_path+cell_line+'/training.csv.gz',compression='gzip',engine='python',usecols=cols).set_index(['enhancer_name', 'promoter_name'])
training_df['old_ind'] = range(len(training_df))

t_pos = training_df[training_df['label']==1]
t_neg = training_df[training_df['label']==0]

## order this df by chromosome in positive and negative obs separately
t_pos.sort_values(by=['enhancer_chrom','enhancer_start','enhancer_end','enhancer_distance_to_promoter','promoter_end', 'promoter_start'],ascending=[True,True,True,True,True,True],inplace=True)

## negative pairs
t_neg.sort_values(by=['enhancer_chrom','enhancer_start','enhancer_end','enhancer_distance_to_promoter','promoter_end', 'promoter_start'],ascending=[True,True,True,True,True,True],inplace=True)
new_df = pd.concat([t_pos,t_neg])
new_df['new_id'] = range(len(new_df))
new_old=pd.Series(new_df.old_ind).tolist()
#==============================================================================
# new_old.npy
#==============================================================================
# the re-ordered epigenomic data; new_old is the index of the original epigenomic dataset in the ordered epigenomic (going to be matched with sequence)
np.save(file_path+cell_line+'/new_old',new_old)


## find out the chromosome order in the ordered epigenomic data (which has the same order as our re-ordered sequence)
chromls = new_df['enhancer_chrom'].unique().tolist()
#==============================================================================
# save chromls.npy
#==============================================================================
np.save(file_path+'a_cum',chromls)

## a_cum is the dict for cumulative counts in each chromosome in positive pairs
## b_cum is the dict for cumulative counts in each chromosome in negative pairs
a=dict((chrom,0) for chrom in chromls)
for chrom in chromls:
    a[chrom]=new_df[(new_df['enhancer_chrom']==chrom) & (new_df['label']==1)].shape[0]

b=dict((chrom,0) for chrom in chromls)
for chrom in chromls:
    b[chrom]=new_df[(new_df['enhancer_chrom']==chrom) & (new_df['label']==0)].shape[0]

a1=[a[chrom] for chrom in chromls]
a_cum=list(accumulate(a1))
b1=[b[chrom] for chrom in chromls]
b_cum=list(accumulate(b1))

file_path = file_path+'/K562/'

#==============================================================================
#  a_cum.npy
#  b_cum.npy
#==============================================================================
np.save(file_path+'a_cum',a_cum)
np.save(file_path+'b_cum',b_cum)




#==============================================================================
# seq_order.npy
#==============================================================================
file_path='/panfs/roc/groups/1/panwei/xiaox345/data/targetfinder/Data/'
cell_line='K562'
# the dict is in a format where epigenomic index:sequence index
with open(file_path+cell_line+'/K562_allmatchdict3.pkl','rb') as f: mykey=pickle.load(f)


a=mykey.keys()
# make sure the order of the key is in [0,1,2,...,41476]
# sum(1 if x==y else 0 for x,y in zip(a,range(41477)))==41477
d=[mykey[x] for x in a] 
del d[d.index(41476)] 
#==============================================================================
# seq_order.npy
#==============================================================================
np.save(file_path+'Seq_order',d)
