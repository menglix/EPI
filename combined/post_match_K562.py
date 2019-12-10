
import pandas as pd
from itertools import compress


import numpy as np
import cPickle


def comparels(original_seq):
    '''
    The function returns a missed observation infered by the context of nearby matched sequence index 
    1. find the consective sequence of numbers (num_seq) from nearby sequence or original 
    2. find the difference between the num_seq and the original sequence
        if they only differ by 1 number, then reassign the missed value to the unmatched epigenomics key; else, find the closest number in the missed sequence or return []
    '''
    min_seq = min(original_seq)
    max_seq = min_seq+20
    num_seq=range(min_seq,max_seq)
    diff = list(set(num_seq)-set(original_seq))
    if len(diff)==1:
        v = diff
    elif len(diff)>1:
        diffls=[]
        for d in diff:
            short_seq=[]
            for k in range(key-3,key+3):
                if k in data1.keys():
                    short_seq.append(data1[k])
            short_seq1=[item for sublist in short_seq for item in sublist]
            short_seq2=reject_outliers(short_seq1)
            total_diff=sum([x-d for x in short_seq2])
            diffls.append(abs(total_diff))
        a=diffls.index(min(diffls))
        v = [diff[a]]
    else:
        v =[]
    return v

def reject_outliers(data):
    '''
    Find the outlier from a given number sequence
    '''
    return list(compress(data,abs(data - np.median(data)) < 20))

def calf(index):
    '''
    The function return a list of matched pair between epigenomic index and sequence index (nearby unmatched sequence)
    mls is the unmatchedd key and we search around it to find what sequences nearby epigenomic data is linked to
    return the matched value (data2) with a given key
    '''
    for i in range(mls[index]-10,mls[index]+10):
        if i in data2.keys():
            print i, data2[i]
    print a[index]



cell_line = 'K562'
data_path = '/data/data/TargetFinder/'
cols = ['label','enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'enhancer_name', 'promoter_name','enhancer_distance_to_promoter']
training_df = pd.read_csv('/home/menglix/data/targetfinder/'+cell_line+'/training.csv.gz',compression='gzip',engine='python',usecols=cols).set_index(['enhancer_name', 'promoter_name'])
t_pos = training_df[training_df['label']==1]
t_neg = training_df[training_df['label']==0]
num_pos_pairs = t_pos.shape[0]
with open('/data/data/TargetFinder/'+cell_line+'/'+cell_line+'_nomatchdict.pkl','rb')as f: data=cPickle.load(f)

with open('/data/data/TargetFinder/'+cell_line+'/'+cell_line+'_matchdict.pkl','rb')as f: data1=cPickle.load(f)

with open('/data/data/TargetFinder/'+cell_line+'/'+cell_line+'_matchdict.pkl','rb')as f: data2=cPickle.load(f)

t_neg.sort_values(by=['enhancer_chrom','enhancer_start','enhancer_end','enhancer_distance_to_promoter','promoter_end', 'promoter_start'],ascending=[True,True,True,True,True,True],inplace=True)
t_pos = training_df[training_df['label']==1]
t_pos.sort_values(by=['enhancer_chrom','enhancer_start','enhancer_end','enhancer_distance_to_promoter','promoter_end', 'promoter_start'],ascending=[True,True,True,True,True,True],inplace=True)


keylist = data.keys()
sortkey=keylist.sort()
match = {}



#find the consective sequence nearby an epigenomic index and update matched dictionary (data1)
for key in keylist:
    original_seq=[]
    for j in range(key-10,key+10):
        if j in data1.keys():
            original_seq.append(data1[j])
    original_seq0=[item for sublist in original_seq for item in sublist]
    original_seq2=reject_outliers(original_seq0)
    a=comparels(original_seq2)
    if a!=[]:
        match[key]=a
    else:
        print key
    data1.update(match)



match_finalkey = data1.keys()
sort_m = match_finalkey.sort()
valuels=[]
value = {}
#construct 1-1 relationship
#if the dictionary has multiple match, return the minimum value of the sequence index that haven't been matched before
for j in match_finalkey:
    if len(data1[j]) == 1:
        ## valuels stores all values occured previously
        valuels.append(data1[j][0])
        value[j] = data1[j][0]
    else:
        u = []
        u = [v for v in data1[j] if v not in valuels]
        value[j] = min(u)
        valuels.append(value[j])

#look at the nearby observation
for i in range(j-20,j+10):
    if i in data1.keys():
        print i, data1[i]

value[1922]=1922
for j in match_finalkey:
    if j > 1922:
        if len(data1[j]) == 1:
            ## valuels stores all values occured previously
            valuels.append(data1[j][0])
            value[j] = data1[j][0]
        else:
            u = []
            u = [v for v in data1[j] if v not in valuels]
            value[j] = min(u)
            valuels.append(value[j])
            
            
            
for i in range(j-20,j+10):
    if i in data1.keys():
        print i, data1[i]
value[26045]=26962
value[26047]=26964

for j in match_finalkey:
    if j > 26047:
        if len(data1[j]) == 1:
            ## valuels stores all values occured previously
            valuels.append(data1[j][0])
            value[j] = data1[j][0]
        else:
            u = []
            u = [v for v in data1[j] if v not in valuels]
            value[j] = min(u)
            valuels.append(value[j])
            
#find which sequence index has not been matched yet
inv_map = {v: k for k, v in value.iteritems()}
a=[x for x in range(41477) if x not in inv_map.keys()]
## mls contains all sequence index not matched
mls=[]
for m in a:
    mls.append(inv_map[m-1])
    
### manual inspection from the context
calf(0)
value[1911]=a[0]
calf(1)
value[23457]=a[1]
calf(2)
value[24996]=a[2]
calf(3)
calf(4)
value[5599]=a[3]
value[5601]=a[4]
calf(5)
value[5810]=a[5]
calf(6)
value[35438]=a[6]
calf(7)
value[12395]=a[7]
calf(8)
value[30256]=a[8]
calf(9)
value[30971]=a[9]



#==============================================================================
# next step
#==============================================================================
mykey = value


a=list(set(mykey.values()))
len(a)==41477
inv_map = {v: k for k, v in mykey.iteritems()}

a=[x for x in range(41477) if x not in inv_map.keys()]
a
mls=[]
for m in a:
    mls.append(inv_map[m-1])

mls
def calf(index):
    for i in range(mls[index]-10,mls[index]+10):
        if i in mykey.keys():
            print i, mykey[i]
    print a[index]

calf(0)
mykey[a[0]]=1912

calf(1)
mykey[a[0]]=1912

mykey[39240]=32390
calf(1)
mykey[39712]=32862
calf(2)
mykey[39715]=32867
calf(3)
mykey[28454]=36232
len(set(mykey.values()))
len(set(mykey.keys()))
len(mykey.values)
len(mykey.values())
# count how many times a sequence index occurs in the matched dictionary
c=[mykey.values().count(x) for x in range(41477)]
# Findout which sequence index was matched for not 1 times  (could be 0 or >1)
# 32377 sequence occurs 2 times and 41476 sequence occurs 0 times
d=[i for i in range(41477) if c[i]!=1]
# find the corresponding index of epigenomic data
e = [i for i in range(41477) for x in d if mykey[i]==x] 
[mykey[i] for i in e] # 32377 sequence was matched multiple times 
for i in range(e[0]-10,e[0]+10): print mykey[i]

for i in range(e[1]-10,e[1]+10): print mykey[i]
# 20776 epigenomic has the same match with 39227 epigenomics data, but it occurs in the end of the consecutive number so match it with the sequence 41476
mykey[e[0]]=41476
# sanity check
len(set(mykey.values()))
with open(data_path + cell_line + '/' + cell_line + '_allmatchdict3.pkl', 'wb') as f: cPickle.dump(mykey, f)
