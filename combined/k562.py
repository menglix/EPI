import pandas as pd
import numpy as np
from lxml import etree
import requests
import cPickle



def getEPseq(training_df,seg,i):
    '''
    Obtain the DNA sequence from hg19 if we know the start, end and chromosome information from the bed files
    seg: enhancer/promoter
    i: index of the pair in the bed or training_df
    '''
    chr = training_df[seg+"_chrom"][i]
    ## 1-based
    start = training_df[seg+"_start"][i]+1
    end = training_df[seg+"_end"][i]
    data = requests.get("http://genome.ucsc.edu/cgi-bin/das/hg19/dna?segment="+ chr + ":"+str(start)+","+str(end),stream=True)
    data.raw.decode_content=True
    a=etree.parse(data.raw)
    b=(etree.tostring(a,method="text")).replace('\n','')
    return b


def accumulate(lis):
    total = 0
    for item in lis:
        total += item
        yield total






cell_lines = ['K562']
data_path = '/data/data/TargetFinder/'
out_path = data_path
for cell_line in cell_lines:
    SPEID = {}
    SPEID['enhancer'] = pd.read_csv(out_path + cell_line + '/' + cell_line + '_' +'enhancer'+ '.csv')
    SPEID['promoter'] = pd.read_csv(out_path + cell_line + '/' + cell_line + '_' +'promoter'+ '.csv')
    cols = ['label','enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'enhancer_name', 'promoter_name','enhancer_distance_to_promoter']
    training_df = pd.read_csv('/home/menglix/data/targetfinder/'+cell_line+'/training.csv.gz',compression='gzip',engine='python',usecols=cols).set_index(['enhancer_name', 'promoter_name'])
    t_pos = training_df[training_df['label']==1]
    t_neg = training_df[training_df['label']==0]
    
    
    num_pos_pairs = t_pos.shape[0]
    t_pos.sort_values(by=['enhancer_chrom','enhancer_start','enhancer_end','enhancer_distance_to_promoter','promoter_end', 'promoter_start'],ascending=[True,True,True,True,True,True],inplace=True)
    chr_pos_count=t_pos.groupby(by=['enhancer_chrom']).count()[[0]].values.tolist()
    chr_pos_countls = [c[0] for c in chr_pos_count]
    index=list(accumulate(chr_pos_countls))
    index.append(0)
    match = {}
    no_match = {}
    for chr in range(23):
        print ('processing chr'+str(chr)+' in positive pairs')
        num_pos_pairs_per_chr = chr_pos_countls[chr]
        t_chr = t_pos.iloc[index[chr-1]:index[chr]]
        for i in range(index[chr-1],index[chr]):
            eplist = {}
            for seg in ['enhancer','promoter']:
                # get the sequence by enhancer or promoter start and end location
                b = getEPseq(t_chr, seg, i-index[chr-1])
                if len(b) < len(SPEID[seg]['words'][0]):
                    eplist[seg+'_list'] = SPEID[seg][index[chr-1]:index[chr]].index[SPEID[seg]['words'][index[chr-1]:index[chr]].str.contains('\w'+b+'\w',case=False)].tolist()
                else:
                    c = b.upper()
                    mask = [c.find(x) for x in SPEID[seg]['words'][index[chr-1]:index[chr]]]
                    ### plus sth
                    eplist[seg+'_list'] = [z+index[chr-1] for z,y in enumerate(mask) if y >= 0]
            if list(set(eplist['enhancer_list']).intersection(eplist['promoter_list'])) == []:
                no_match[i] = cell_line+seg
            else:
                match[i] = np.intersect1d(eplist['enhancer_list'],eplist['promoter_list'], assume_unique=True)

    with open(out_path + cell_line + '/' + cell_line + '_pos_matchdict.pkl', 'wb') as f:
        cPickle.dump(match, f)
    with open(out_path + cell_line + '/' + cell_line + '_pos_nomatchdict.pkl', 'wb') as f2:
        cPickle.dump(no_match, f2)



    ## negative pairs
    num_neg_pairs = t_neg.shape[0]
    t_neg.sort_values(by=['enhancer_chrom','enhancer_start','enhancer_end','enhancer_distance_to_promoter','promoter_end', 'promoter_start'],ascending=[True,True,True,True,True,True],inplace=True)
    chr_neg_count=t_neg.groupby(by=['enhancer_chrom']).count()[[0]].values.tolist()
    chr_neg_countls = [c[0] for c in chr_neg_count]
    index = None
    index=list(accumulate(chr_neg_countls))
    index.append(0)

    for chr in range(23):
        print ('processing chr'+str(chr)+' in negative pairs')
        num_neg_pairs_per_chr = chr_neg_countls[chr]
        t_chr = t_neg.iloc[index[chr-1]:index[chr]]
        neg_index = range(num_pos_pairs+index[chr-1],num_pos_pairs+index[chr])
        for i in neg_index:
            eplist = {}
            for seg in ['enhancer','promoter']:
                b = getEPseq(t_chr, seg, i-index[chr-1]-num_pos_pairs)
                if len(b) < len(SPEID[seg]['words'][0]):
                    eplist[seg+'_list'] = SPEID[seg].iloc[num_pos_pairs:].index[SPEID[seg]['words'][num_pos_pairs:].str.contains('\w'+b+'\w',case=False)].tolist()
                else:
                    c = b.upper()
                    mask = [c.find(x) for x in SPEID[seg]['words'][num_pos_pairs:]]
                    ### plus sth
                    eplist[seg+'_list'] = [z+num_pos_pairs for z,y in enumerate(mask) if y >= 0]
            if list(set(eplist['enhancer_list']).intersection(eplist['promoter_list'])) == []:
                no_match[i] = cell_line+seg
            else:
                match[i] = np.intersect1d(eplist['enhancer_list'],eplist['promoter_list'], assume_unique=True)

    with open(out_path + cell_line + '/' + cell_line + '_matchdict.pkl', 'wb') as f:
        cPickle.dump(match, f)
    with open(out_path + cell_line + '/' + cell_line + '_nomatchdict.pkl', 'wb') as f2:
        cPickle.dump(no_match, f2)
        

    