#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit
import heapq

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import accuracy_score, average_precision_score, precision_score,recall_score

from conf import *

import DataReader
import evaluation
import net as network
import utils
import performance

from document import *

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, "PID", os.getpid()

torch.cuda.set_device(args.gpu)

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n): 
        mcp[i].data[:] = mp[i].data[:]
 
def main():

    DIR = args.DIR
    embedding_file = args.embedding_dir

    best_network_file = "./model/network_model_pretrain.best.top"
    print >> sys.stderr,"Read model from ",best_network_file
    best_network_model = torch.load(best_network_file)

    embedding_matrix = numpy.load(embedding_file)
    "Building torch model"
    worker = network.Network(nnargs["pair_feature_dimention"],nnargs["mention_feature_dimention"],nnargs["word_embedding_dimention"],nnargs["span_dimention"],1000,nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix).cuda()
    net_copy(worker,best_network_model)

    best_network_file = "./model/network_model_pretrain.best.top"
    print >> sys.stderr,"Read model from ",best_network_file
    best_network_model = torch.load(best_network_file)

    manager = network.Network(nnargs["pair_feature_dimention"],nnargs["mention_feature_dimention"],nnargs["word_embedding_dimention"],nnargs["span_dimention"],1000,nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix).cuda()
    net_copy(manager,best_network_model)

    reduced=""
    if args.reduced == 1:
        reduced="_reduced"

    print >> sys.stderr,"prepare data for train ..."
    train_docs_iter = DataReader.DataGnerater("train"+reduced)
    print >> sys.stderr,"prepare data for dev and test ..."
    dev_docs_iter = DataReader.DataGnerater("dev"+reduced)
    test_docs_iter = DataReader.DataGnerater("test"+reduced)

    print "Performance after pretraining..."
    print "DEV"
    metric = performance.performance(dev_docs_iter,worker,manager) 
    print "Average:",metric["average"]
    print "TEST"
    metric = performance.performance(test_docs_iter,worker,manager) 
    print "Average:",metric["average"]
    print "***"
    print
    sys.stdout.flush()

    lr = 0.000002
    dropout_rate = 0.5
    times = 0

    model_save_dir = "./model/reinforce/"
    utils.mkdir(model_save_dir)

    score_softmax = nn.Softmax()

    optimizer_manager = optim.RMSprop(manager.parameters(), lr=lr, eps = 1e-5)
    optimizer_worker = optim.RMSprop(worker.parameters(), lr=lr, eps = 1e-5)
   
    for echo in range(30):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        train_docs = utils.load_pickle(args.DOCUMENT + 'train_docs.pkl')
        docs_by_id = {doc.did: doc for doc in train_docs}

        train_docs_m = utils.load_pickle(args.DOCUMENT + 'train_docs.pkl')
        docs_by_id_manager = {doc.did: doc for doc in train_docs_m}
       
        print >> sys.stderr,"Link docs ..."
        tmp_data = []
        cluster_info = {0:[0]}
        cluster_list = [0]
        current_new_cluster = 1
        predict_action_embedding = []
        choose_action = []
        mid = 1
        for data in train_docs_iter.rl_case_generater(shuffle=True):
            mention_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
            mention_spans = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
            candi_index = autograd.Variable(torch.from_numpy(data["candi_word_index"]).type(torch.cuda.LongTensor))
            candi_spans = autograd.Variable(torch.from_numpy(data["candi_span"]).type(torch.cuda.FloatTensor))
            pair_feature = autograd.Variable(torch.from_numpy(data["pair_features"]).type(torch.cuda.FloatTensor))
            anaphors = autograd.Variable(torch.from_numpy(data["pair_anaphors"]).type(torch.cuda.LongTensor))
            antecedents = autograd.Variable(torch.from_numpy(data["pair_antecedents"]).type(torch.cuda.LongTensor))

            anaphoricity_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
            anaphoricity_span = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
            anaphoricity_feature = autograd.Variable(torch.from_numpy(data["anaphoricity_feature"]).type(torch.cuda.FloatTensor))
            rl = data["rl"]

            output_manager, pair_score_manager, mention_pair_representations_manager = manager.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents)
            ana_output_manager, ana_score_manager, ana_pair_representations_manager = manager.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature)

            reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

            scores_manager = torch.transpose(torch.cat((pair_score_manager,ana_score_manager),1),0,1)[reindex]
            representations_manager = torch.cat((mention_pair_representations_manager,ana_pair_representations_manager),0)[reindex]

            for s,e in zip(rl["starts"],rl["ends"]):
                action_embeddings = representations_manager[s:e]

                score = score_softmax(torch.transpose(scores_manager[s:e],0,1)).data.cpu().numpy()[0]
                this_action = utils.choose_action(score)
                if this_action == len(score)-1 :
                    ana_score = 1.0
                    should_cluster = current_new_cluster
                    cluster_info[should_cluster] = []
                    current_new_cluster += 1
                else:
                    ana_score = 0.0 
                    should_cluster = cluster_list[this_action]

                choose_action.append(this_action)
                cluster_info[should_cluster].append(mid)
                cluster_list.append(should_cluster)
                mid += 1

                cluster_indexs = torch.cuda.LongTensor(cluster_info[should_cluster])
                action_embedding_choose = torch.mean(action_embeddings[cluster_indexs],0,keepdim=True)
                predict_action_embedding.append(action_embedding_choose)

            # get action_embedding
            tmp_data.append(data)

            if rl["end"] == True:
                inside_index = 0
                manager_path = []
                worker_path = []
                for data in tmp_data:
                    mention_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
                    mention_spans = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
                    candi_index = autograd.Variable(torch.from_numpy(data["candi_word_index"]).type(torch.cuda.LongTensor))
                    candi_spans = autograd.Variable(torch.from_numpy(data["candi_span"]).type(torch.cuda.FloatTensor))
                    pair_feature = autograd.Variable(torch.from_numpy(data["pair_features"]).type(torch.cuda.FloatTensor))
                    anaphors = autograd.Variable(torch.from_numpy(data["pair_anaphors"]).type(torch.cuda.LongTensor))
                    antecedents = autograd.Variable(torch.from_numpy(data["pair_antecedents"]).type(torch.cuda.LongTensor))

                    anaphoricity_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
                    anaphoricity_span = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
                    anaphoricity_feature = autograd.Variable(torch.from_numpy(data["anaphoricity_feature"]).type(torch.cuda.FloatTensor))
                    rl = data["rl"]

                    output_worker, pair_score_worker, mention_pair_representations_worker = worker.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents)
                    ana_output_worker, ana_score_worker, ana_pair_representations_worker = worker.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature)

                    reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

                    scores_worker = torch.transpose(torch.cat((pair_score_worker,ana_score_worker),1),0,1)[reindex]
                    representations_worker = torch.cat((mention_pair_representations_worker,ana_pair_representations_worker),0)[reindex]

                    doc = docs_by_id[rl["did"]] 
                    doc_manager = docs_by_id_manager[rl["did"]]
                    for s,e in zip(rl["starts"],rl["ends"]):
                        action_embeddings = representations_worker[s:e]
                        score = score_softmax(torch.transpose(scores_worker[s:e],0,1)).data.cpu().numpy()[0]
                    
                        action_embedding_choose = predict_action_embedding[inside_index]
                        similarities = torch.sum(torch.abs(action_embeddings - action_embedding_choose),1)
                        similarities = similarities.data.cpu().numpy()
    
                        ## choose action from similarities
                        ## choose the last n smallest actions

                        action_probabilities = []
                        action_list = []
                        action_candidates = heapq.nlargest(5,-similarities)
                        for action in action_candidates:
                            action_index = numpy.argwhere(similarities == -action)[0][0]
                            action_probabilities.append(score[action_index])
                            action_list.append(action_index)

                        sample_action = utils.sample_action(numpy.array(action_probabilities))

                        manager_action = choose_action[inside_index]
                        worker_action = action_list[sample_action]
                    
                        inside_index += 1

                        link = worker_action
                        m1, m2 = rl['ids'][s + link]
                        doc.link(m1, m2)

                        #link = manager_action
                        #m1, m2 = rl['ids'][s + link]
                        #doc_manager.link(m1, m2)

                        manager_path.append(manager_action)
                        worker_path.append(worker_action)

                #manager_path = autograd.Variable(torch.cuda.LongTensor(manager_path))
                #worker_path = autograd.Variable(torch.cuda.LongTensor(worker_path))
                inside_index = 0
                for data in tmp_data:
                    rl = data["rl"]
                    reward = -1.0*docs_by_id[rl["did"]].get_f1()

                    mention_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
                    mention_spans = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
                    candi_index = autograd.Variable(torch.from_numpy(data["candi_word_index"]).type(torch.cuda.LongTensor))
                    candi_spans = autograd.Variable(torch.from_numpy(data["candi_span"]).type(torch.cuda.FloatTensor))
                    pair_feature = autograd.Variable(torch.from_numpy(data["pair_features"]).type(torch.cuda.FloatTensor))
                    anaphors = autograd.Variable(torch.from_numpy(data["pair_anaphors"]).type(torch.cuda.LongTensor))
                    antecedents = autograd.Variable(torch.from_numpy(data["pair_antecedents"]).type(torch.cuda.LongTensor))

                    anaphoricity_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
                    anaphoricity_span = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
                    anaphoricity_feature = autograd.Variable(torch.from_numpy(data["anaphoricity_feature"]).type(torch.cuda.FloatTensor))

                    output_manager, pair_score_manager, mention_pair_representations_manager = manager.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,nnargs["dropout_rate"])
                    ana_output_manager, ana_score_manager, ana_pair_representations_manager = manager.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature,nnargs["dropout_rate"])

                    reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

                    scores_manager = torch.transpose(torch.cat((pair_score_manager,ana_score_manager),1),0,1)[reindex]
                    representations_manager = torch.cat((mention_pair_representations_manager,ana_pair_representations_manager),0)[reindex]
                    optimizer_manager.zero_grad
                    manager_loss = None
                    for s,e in zip(rl["starts"],rl["ends"]):
                        score = score_softmax(torch.transpose(scores_manager[s:e],0,1))
                        this_cost = score[:,manager_path[inside_index]]*reward
                        if manager_loss is None:
                            manager_loss = this_cost
                        else:
                            manager_loss += this_cost
                    manager_loss.backward()
                    optimizer_manager.step()

                    mention_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
                    mention_spans = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
                    candi_index = autograd.Variable(torch.from_numpy(data["candi_word_index"]).type(torch.cuda.LongTensor))
                    candi_spans = autograd.Variable(torch.from_numpy(data["candi_span"]).type(torch.cuda.FloatTensor))
                    pair_feature = autograd.Variable(torch.from_numpy(data["pair_features"]).type(torch.cuda.FloatTensor))
                    anaphors = autograd.Variable(torch.from_numpy(data["pair_anaphors"]).type(torch.cuda.LongTensor))
                    antecedents = autograd.Variable(torch.from_numpy(data["pair_antecedents"]).type(torch.cuda.LongTensor))

                    anaphoricity_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
                    anaphoricity_span = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
                    anaphoricity_feature = autograd.Variable(torch.from_numpy(data["anaphoricity_feature"]).type(torch.cuda.FloatTensor))

                    output_worker, pair_score_worker, mention_pair_representations_worker = worker.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,nnargs["dropout_rate"])
                    ana_output_worker, ana_score_worker, ana_pair_representations_worker = worker.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature,nnargs["dropout_rate"])

                    reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

                    scores_worker = torch.transpose(torch.cat((pair_score_worker,ana_score_worker),1),0,1)[reindex]
                    representations_worker = torch.cat((mention_pair_representations_worker,ana_pair_representations_worker),0)[reindex]

                    optimizer_worker.zero_grad
                    worker_loss = None
                    for s,e in zip(rl["starts"],rl["ends"]):
                        score = score_softmax(torch.transpose(scores_worker[s:e],0,1))
                        this_cost = score[:,worker_path[inside_index]]*reward
                        if worker_loss is None:
                            worker_loss = this_cost
                        else:
                            worker_loss += this_cost
                    worker_loss.backward()
                    optimizer_worker.step()

                    inside_index += 1 

                tmp_data = []
                cluster_info = {0:[0]}
                cluster_list = [0]
                current_new_cluster = 1
                mid = 1
                predict_action_embedding = []
                choose_action = []

        end_time = timeit.default_timer()

        print >> sys.stderr, "TRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr,"save model ..."
        #torch.save(network_model, model_save_dir+"network_model_rl_worker.%d"%echo)
        #torch.save(ana_network, model_save_dir+"network_model_rl_manager.%d"%echo)
        
        print "DEV"
        metric = performance.performance(dev_docs_iter,worker,manager) 
        print "Average:",metric["average"]
        print "DEV Ana: ",metric["ana"]
        print "TEST"
        metric = performance.performance(test_docs_iter,worker,manager) 
        print "Average:",metric["average"]
        print "TEST Ana: ",metric["ana"]
        print
        sys.stdout.flush()

if __name__ == "__main__":
    main()
