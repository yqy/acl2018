#coding=utf8
import sys
import numpy
import numpy as np
from tempfile import TemporaryFile
import random

import timeit
import cPickle
import json

from document import *

from conf import *
import utils

MENTION_TYPES = {
    "PRONOMINAL": 0,
    "NOMINAL": 1,
    "PROPER": 2,
    "LIST": 3
}
MENTION_NUM, SENTENCE_NUM, START_INDEX, END_INDEX, MENTION_TYPE, CONTAINED = 0, 1, 2, 3, 4, 5

DIR = args.DIR
embedding_file = DIR+"features/mention_data/word_vectors.npy"

numpy.set_printoptions(threshold=numpy.nan)
random.seed(args.random_seed)

class DataGnerater():
    def __init__(self,file_name):

        doc_path = DIR+"features/doc_data/%s/"%file_name
        pair_path = DIR+"features/mention_pair_data/%s/"%file_name
        mention_path = DIR+"features/mention_data/%s/"%file_name

        gold_path = DIR+"gold/"+file_name.split("_")[0]
        # read gold chain
        self.gold_chain = {}
        gold_file = open(gold_path)
        golds = gold_file.readlines()
        for item in golds:
            gold = json.loads(item) 
            self.gold_chain[int(gold.keys()[0])] = list(gold[gold.keys()[0]])


    ## for mentions
        self.mention_spans = numpy.load(mention_path+"msp.npy")
        self.mention_word_index = numpy.load(mention_path+"mw.npy")[:, :-1] 
        self.mention_feature = numpy.load(mention_path+"mf.npy")
        self.mention_id_raw = numpy.load(mention_path+"mid.npy")
        self.mention_id = numpy.load(mention_path+"mid.npy")[:,0]
        self.mention_did = numpy.load(mention_path+"mdid.npy")[:,0]
        self.mention_num = numpy.load(mention_path+"mnum.npy")[:,0]

        self.mention_pair_feature = numpy.load(mention_path+"yqy.npy",mmap_mode='r')

        #self.mention_feature_arrays = [] ## mention feature is saved in this array
        self.mention_feature_arrays = numpy.load(mention_path+"mfyqy.npy",mmap_mode='r')

    ## for pairs
        #self.pair_feature = numpy.load(pair_path + 'pf.npy')
        self.pair_coref_info = numpy.load(pair_path + "y.npy")    
        self.pair_index = numpy.load(pair_path + "pi.npy")
        self.pair_mention_id = numpy.load(pair_path + "pmid.npy")

    ## for docs
        self.document_features = numpy.load(doc_path + 'df.npy')
        self.doc_pairs = numpy.load(doc_path + 'dpi.npy') # each line is the pair_start_index -- pair_end_index
        self.doc_mentions = numpy.load(doc_path + 'dmi.npy') # each line is the mention_start_index -- mention_end_index

        self.batch = []
        self.doc_batch = {}
        # build training data  
        doc_index = range(self.doc_pairs.shape[0])
        done_num = 0
        total_num = self.doc_pairs.shape[0]
        estimate_time = 0.0
        self.n_anaphors = 0
        self.n_pairs = 0

        for did in doc_index:
            start_time = timeit.default_timer() 
            ps, pe = self.doc_pairs[did]
            ms, me = self.doc_mentions[did]
            self.n_anaphors += me - ms

            self.doc_batch[did] = []
            
            done_num += 1

            doc_mention_sizes = me - ms
            document_feature = self.document_features[did]

            max_pairs = 10000
            min_anaphor = 1
            min_pair = 0
            while min_anaphor < doc_mention_sizes:
                max_anaphor = min(new_max_anaphor(min_anaphor, max_pairs), me - ms)
                max_pair = min(max_anaphor * (max_anaphor - 1) / 2, pe - ps) 

                mentions = np.arange(ms, ms + max_anaphor)
                antecedents = np.arange(max_anaphor - 1)
                anaphors = np.arange(min_anaphor, max_anaphor)
                pairs = np.arange(ps + min_pair, ps + max_pair)
                pair_antecedents = np.concatenate([np.arange(ana)
                                                    for ana in range(min_anaphor, max_anaphor)]) 
                pair_anaphors = np.concatenate([(ana - min_anaphor) *
                                                    np.ones(ana, dtype='int32')
                                                    for ana in range(min_anaphor, max_anaphor)])
                '''
                print "mentions",len(mentions),mentions # 表示应该取的mention list
                print "ante",len(antecedents),antecedents #表示每个batch (mention list) 之中,要选取作为antecedents的index
                print "anaph",len(anaphors),anaphors #表示每个batch (mention list) 之中,要选取作为anaphors的index
                print "pairs",len(pairs),pairs #表示对应的pair_feature之类的信息
                print "pair_ante",len(pair_antecedents),pair_antecedentsi #表示要变成pair的antecedents index
                print "pair_ana",len(pair_anaphors),pair_anaphors #表示要变成pair的ana index
                '''


                positive, negative = [], []
                ana_to_pos, ana_to_neg = {}, {}

                ys = self.pair_coref_info[pairs]
                for i, (ana, y) in enumerate(zip(pair_anaphors, ys)):
                    labels = positive if y == 1 else negative
                    ana_to_ind = ana_to_pos if y == 1 else ana_to_neg
                    if ana not in ana_to_ind:
                        ana_to_ind[ana] = [len(labels), len(labels)]
                    else:
                        ana_to_ind[ana][1] = len(labels)
                    labels.append(i)

                # positive : index of positive examples in pairs
                # negative : index of negative examples in pairs

                pos_starts, pos_ends, neg_starts, neg_ends = [], [], [], []
                anaphoricities = []
                for ana in range(0, max_anaphor - min_anaphor):
                    if ana in ana_to_pos:
                        start, end = ana_to_pos[ana]
                        pos_starts.append(start)
                        pos_ends.append(end + 1)
                        anaphoricities.append(1)
                    else:
                        anaphoricities.append(0)
                    if ana in ana_to_neg:
                        start, end = ana_to_neg[ana]
                        neg_starts.append(start)
                        neg_ends.append(end + 1)
                # anaphoricities: = 1 if mention is anaphoricities else 0

                starts, ends = [], []
                costs = []
                reindex = []
                pair_pos, anaphor_pos = 0, len(pairs)
                i, j = 0, 0
                for ana in range(0, max_anaphor - min_anaphor):
                    ana_labels = []
                    ana_reindex = []
                    start = i 
                    for ant in range(0, ana + min_anaphor):
                        ana_labels.append(ys[j])
                        i += 1
                        j += 1
                        ana_reindex.append(pair_pos)
                        pair_pos += 1
                    i += 1
                    ana_reindex.append(anaphor_pos)
                    anaphor_pos += 1

                    end = i 
                    ana_labels = np.array(ana_labels)
                    anaphoric = ana_labels.sum() > 0 
                    if end > (start + 1):
                        starts.append(start)
                        ends.append(end)
                        reindex += ana_reindex
                    else:
                        i = start
                        continue 

                    WL = 0.0
                    FL = 0.0
                    FN = 0.0

                    if anaphoric:
                        ana_costs = np.append(WL * (ana_labels ^ 1), FN)
                    else:
                        ana_costs = np.append(FL * np.ones_like(ana_labels), 0)
                    costs += list(ana_costs)

                positive = numpy.array(positive,dtype='int32')
                negative = numpy.array(negative,dtype='int32')
                pos_starts = np.array(pos_starts, dtype='int32')
                pos_ends = np.array(pos_ends, dtype='int32')
                neg_starts = np.array(neg_starts, dtype='int32')
                neg_ends = np.array(neg_ends, dtype='int32')
                reindex = np.array(reindex, dtype='int32')
                costs = np.array(costs, dtype='float')

                rl = {}
                rl["starts"] = numpy.array(starts,dtype='int32')
                rl["ends"] = numpy.array(ends,dtype='int32')
                rl["did_num"] = did
                rl["reindex"] = reindex
                rl["costs"] = costs

                anaphor_ids = self.mention_id_raw[mentions][anaphors]
                antecedent_ids = self.mention_id_raw[mentions][antecedents]
                pair_antecedent_ids = antecedent_ids[pair_antecedents]
                pair_anaphor_ids = anaphor_ids[pair_anaphors]
                pair_ids = np.hstack([pair_antecedent_ids, pair_anaphor_ids])
                anaphor_ids = np.hstack([-1 * np.ones_like(anaphor_ids), anaphor_ids])
                all_ids = np.vstack([pair_ids, anaphor_ids])
                rl['ids'] = all_ids[reindex]
                rl['did'] = self.pair_mention_id[pairs][0, 0]

                data = {}
                #data["mentions"] = mentions
                #data["antecedents"] = antecedents
                #data["anaphors"] = anaphors
                #data["pairs"] = pairs


                data["pair_antecedents"] = pair_antecedents
                data["pair_anaphors"] = pair_anaphors
                data["positive"] = positive
                data["negative"] = negative
                data["top_score_index"] = np.concatenate([positive, negative])
                data["top_starts"] = np.concatenate([pos_starts, positive.size + neg_starts])
                data["top_ends"] = np.concatenate([pos_ends, positive.size + neg_ends])
                data["top_gold"] = np.concatenate([np.ones(pos_starts.size),np.zeros(neg_starts.size)])
                data["rl"] = rl

                data["candi_word_index"] = self.mention_word_index[mentions[0]:mentions[-1]+1][antecedents]
                data["candi_span"] = self.mention_spans[mentions[0]:mentions[-1]+1][antecedents]
                data["candi_ids"] = self.mention_id[mentions[0]:mentions[-1]+1][antecedents]
                data["candi_ids_all"] = self.mention_id[mentions[0]:mentions[-1]+1]

                data["mention_word_index"] = self.mention_word_index[mentions[0]:mentions[-1]+1][anaphors]
                data["mention_span"] = self.mention_spans[mentions[0]:mentions[-1]+1][anaphors]

                data["pair_features"] = self.mention_pair_feature[pairs[0]:pairs[-1]+1]
                data["pair_target"] = self.pair_coref_info[pairs[0]:pairs[-1]+1].astype(int)

                data["anaphoricity_feature"] = self.mention_feature_arrays[mentions[0]:mentions[-1]+1][anaphors]
                data["anaphoricity_target"] = numpy.array(anaphoricities)

                self.batch.append(data)
                self.doc_batch[did].append(data)

                min_anaphor = max_anaphor
                min_pair = max_pair
                self.n_pairs += len(pairs)

            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
        self.anaphors_per_batch = float(self.n_anaphors) / float(len(self.batch))
        self.pairs_per_batch = float(self.n_pairs)/float(len(self.batch))

        self.scale_factor = self.pairs_per_batch
        self.anaphoricity_scale_factor = 50 * self.anaphors_per_batch 

        self.scale_factor_top = 10*self.anaphors_per_batch
        self.anaphoricity_scale_factor_top = 20 * self.anaphors_per_batch 

    def train_generater(self,filter_num=700,shuffle=False):
        if shuffle:
            numpy.random.shuffle(self.batch) 

        done_num = 0
        total_num = len(self.batch)
        estimate_time = 0.0
        for data in self.batch:
            start_time = timeit.default_timer() 
            done_num += 1
            yield data

            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            info = "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)
            sys.stderr.write(info+"\r")
        print >> sys.stderr

    def rl_case_generater(self,shuffle=False):

        index_list = range(len(self.doc_batch.keys()))

        if shuffle:
            random.shuffle(index_list) 

        done_num = 0
        total_num = len(self.doc_batch)
        estimate_time = 0.0

        for did_index in index_list:#[905:]:
            start_time = timeit.default_timer() 
            did = self.doc_batch.keys()[did_index]
            done_num += 1
            
            i = 0
            for data in self.doc_batch[did]:
                i += 1

                data["rl"]["end"] = False
                if i == len(self.doc_batch[did]):
                    data["rl"]["end"] = True
                yield data
            
            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)

            info = "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)
            sys.stderr.write(info+"\r")

        print >> sys.stderr

def new_max_anaphor(n, k): 
    # find m such that sum from i=n to m-1 is < k
    # i.e., total number of pairs with anaphor num between n and m (exclusive) < k
    return max(1, int(np.floor(0.5 * (1 + np.sqrt(8 * k + 4 * n * n - 4 * n + 1))))) 
    
if __name__ == "__main__":
    train_docs = utils.load_pickle(args.DOCUMENT + 'train_docs.pkl')[:3]
    dev_docs = utils.load_pickle(args.DOCUMENT + 'dev_docs.pkl')[:3]

    docs_by_id = {doc.did: doc for doc in train_docs}

    tdata = DataGnerater("train_reduced")   
    #data = DataGnerater("train")   

    for t in tdata.rl_case_generater():
        mention_word_index_return, mention_span_return, candi_word_index_return,candi_span_return,pair_features_return,pair_antecedents,pair_anaphors,target,positive,negative,anaphoricity_word_index, anaphoricity_span, anaphoricity_feature, anaphoricity_target,rl,candi_ids_return = t

        scores = np.ones(len(rl["reindex"]))
        update_doc(docs_by_id[rl['did']], rl, scores)

    for t in tdata.rl_case_generater():
        mention_word_index_return, mention_span_return, candi_word_index_return,candi_span_return,pair_features_return,pair_antecedents,pair_anaphors,target,positive,negative,anaphoricity_word_index, anaphoricity_span, anaphoricity_feature, anaphoricity_target,rl,candi_ids_return = t

        doc = docs_by_id[rl['did']]
        doc_weight = (len(doc.mention_to_gold) + len(doc.mentions)) / 10.0
        for (start, end) in zip(rl['starts'], rl['ends']):
            ids = rl['ids'][start:end]
            ana = ids[0, 1]
            old_ant = doc.ana_to_ant[ana]
            doc.unlink(ana)
            costs = rl['costs'][start:end]
            for ant_ind in range(end - start):
                costs[ant_ind] = doc.link(ids[ant_ind, 0], ana, hypothetical=True, beta=1)
            doc.link(old_ant, ana) 

            costs -= costs.max()
            costs *= -doc_weight
            print "b", rl['costs'][start:end]

