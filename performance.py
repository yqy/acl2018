#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

from sklearn.metrics import accuracy_score, average_precision_score, precision_score,recall_score

from conf import *

import DataReader
import net as network
import evaluation
import utils

import cPickle
sys.setrecursionlimit(1000000)

torch.cuda.set_device(args.gpu)
 
def performance(doc,worker_net,manager_net=None):

    test_document = []

    score_softmax = nn.Softmax()
    
    predict_cluster = []
    new_cluster_num = 1
    predict_cluster.append(0)
    mid = 0
    cluster_info = {0:[0]}

    worker = worker_net
    if manager_net is None:
        manager = worker_net
    else:
        manager = manager_net

    for data in doc.rl_case_generater(shuffle=False):
        
        this_doc = doc
        
        candi_ids_all = data["candi_ids_all"]
        rl = data["rl"]

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

        target = data["pair_target"]
        anaphoricity_target = data["anaphoricity_target"]

        output_manager, pair_score_manager, mention_pair_representations_manager = manager.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents)
        ana_output_manager, ana_score_manager, ana_pair_representations_manager = manager.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature)

        mention_pair_representations_manager = autograd.Variable(torch.from_numpy(mention_pair_representations_manager).type(torch.cuda.FloatTensor))
        ana_pair_representations_manager = autograd.Variable(torch.from_numpy(ana_pair_representations_manager).type(torch.cuda.FloatTensor))

        reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

        scores_manager = torch.transpose(torch.cat((pair_score_manager,ana_score_manager),1),0,1)[reindex]
        representations_manager = torch.cat((mention_pair_representations_manager,ana_pair_representations_manager),0)[reindex]
        
        output_worker, pair_score_worker, mention_pair_representations_worker = worker.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents)
        ana_output_worker, ana_score_worker, ana_pair_representations_worker = worker.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature)
        mention_pair_representations_worker = autograd.Variable(torch.from_numpy(mention_pair_representations_worker).type(torch.cuda.FloatTensor))
        ana_pair_representations_worker = autograd.Variable(torch.from_numpy(ana_pair_representations_worker).type(torch.cuda.FloatTensor))


        scores_worker = torch.transpose(torch.cat((pair_score_worker,ana_score_worker),1),0,1)[reindex]
        representations_worker = torch.cat((mention_pair_representations_worker,ana_pair_representations_worker),0)[reindex]

        for s,e in zip(rl["starts"],rl["ends"]):
            manager_action_embeddings = representations_manager[s:e]
            worker_action_embeddings = representations_worker[s:e]
            #score = score_softmax(torch.transpose(scores_manager[s:e],0,1)).data.cpu().numpy()[0]
            #score = score_softmax(torch.transpose(scores_worker[s:e],0,1)).data.cpu().numpy()[0]
            score = F.softmax(torch.squeeze(scores_worker[s:e]),dim=0).data.cpu().numpy()
            this_action = utils.choose_action(score)
 
            #if this_action == len(score)-1:
            #    cluster_indexs = torch.cuda.LongTensor([this_action])
            #else:
            #    should_cluster = predict_cluster[this_action]
            #    cluster_indexs = torch.cuda.LongTensor(cluster_info[should_cluster]+[this_action])

            #action_embedding_choose = torch.mean(manager_action_embeddings[cluster_indexs],0,keepdim=True)
            #similarities = torch.sum(torch.abs(worker_action_embeddings - action_embedding_choose),1)
            #similarities = similarities.data.cpu().numpy()
            #real_action = numpy.argmin(similarities)

            real_action = this_action
            if real_action == len(score)-1:
                should_cluster = new_cluster_num
                cluster_info[should_cluster] = []
                new_cluster_num += 1
            else:
                should_cluster = predict_cluster[real_action]

            cluster_info[should_cluster].append(mid) 
            predict_cluster.append(should_cluster)
            mid += 1

        if rl["end"] == True:
            ev_document = utils.get_evaluation_document(predict_cluster,this_doc.gold_chain[rl["did"]],candi_ids_all,new_cluster_num)
            test_document.append(ev_document)
            predict_cluster = []
            new_cluster_num = 1
            predict_cluster.append(0)
            cluster_info = {0:[0]}
            mid = 0

    metrics = evaluation.Output_Result(test_document)
    r,p,f = metrics["muc"]
    print "MUC: recall: %f precision: %f  f1: %f"%(r,p,f)
    r,p,f = metrics["b3"]
    print "B3: recall: %f precision: %f  f1: %f"%(r,p,f)
    r,p,f = metrics["ceaf"]
    print "CEAF: recall: %f precision: %f  f1: %f"%(r,p,f)
    print "AVE",metrics["average"]

    return metrics

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]

if __name__ == "__main__":
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

    #dev_docs = DataReader.DataGnerater("dev"+reduced)
    test_docs = DataReader.DataGnerater("test"+reduced)

    metric = performance(test_docs,worker,manager)
    print "Ave",metric["average"]

    #network_file = "./model/network_model_pretrain.top.best"
    #network_model = torch.load(network_file)

    #ana_network_file = "./model/network_model_pretrain.top.best"
    #ana_network_model = torch.load(ana_network_file)

    #reduced=""
    #if args.reduced == 1:
    #    reduced="_reduced"

    #metric = performance(test_docs,network_model,ana_network_model)
    #print "Ave",metric["average"]
