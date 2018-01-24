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
from torch.distributions import Categorical

from tensorboard_logger import configure, log_value, Logger

from sklearn.metrics import accuracy_score, average_precision_score, precision_score,recall_score

from conf import *

import DataReader
import evaluation
import net as network
import utils
import performance
import performance_manager

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
    embedding_matrix = numpy.load(embedding_file)

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
    #train_docs_iter = DataReader.DataGnerater("dev"+reduced)
    print >> sys.stderr,"prepare data for dev and test ..."
    dev_docs_iter = DataReader.DataGnerater("dev"+reduced)
    test_docs_iter = DataReader.DataGnerater("test"+reduced)

    '''
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
    '''

    lr = nnargs["lr"]
    top_k = nnargs["top_k"]*20

    model_save_dir = "./model/reinforce/"
    utils.mkdir(model_save_dir)

    score_softmax = nn.Softmax()

    optimizer_manager = optim.RMSprop(manager.parameters(), lr=lr, eps = 1e-6)

    MAX_AVE = 2048
   
    for echo in range(nnargs["epoch"]):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        reward_log = Logger(Tensorboard+args.tb+"/acl2018/%d/reward/"%echo, flush_secs=3)
        entropy_log_manager = Logger(Tensorboard+args.tb+"/acl2018/%d/entropy/worker"%echo, flush_secs=3)
        entropy_log_worker = Logger(Tensorboard+args.tb+"/acl2018/%d/entropy/manager"%echo, flush_secs=3)

        train_docs = utils.load_pickle(args.DOCUMENT + 'train_docs.pkl')
        #train_docs = utils.load_pickle(args.DOCUMENT + 'dev_docs.pkl')
        docs_by_id = {doc.did: doc for doc in train_docs}

        ave_reward = []
        ave_manager_entropy = []
        ave_worker_entropy = []
       
        print >> sys.stderr,"Link docs ..."
        tmp_data = []
        cluster_info = {0:[0]}
        cluster_list = [0]
        current_new_cluster = 1
        predict_action_embedding = []
        choose_action = []
        mid = 1

        path = []
        
        step = 0

        statistic = {"worker_hits":0, "manager_hits":0, "total":0, "manager_predict_last":0, "worker_predict_last":0}

        for data in train_docs_iter.rl_case_generater(shuffle=True):

            rl = data["rl"]

            scores_manager,representations_manager = get_score_representations(manager,data)

            doc = docs_by_id[rl["did"]]

            for s,e in zip(rl["starts"],rl["ends"]):
                action_embeddings = representations_manager[s:e]

                #score = score_softmax(torch.transpose(scores_manager[s:e],0,1)).data.cpu().numpy()[0]
                #index = utils.choose_action(score)

                probs = score_softmax(torch.transpose(scores_manager[s:e],0,1))
                m = Categorical(probs)
                this_action = m.sample()
                index = this_action.data.cpu().numpy()[0] 

                if index == (e-s-1):
                    should_cluster = current_new_cluster
                    cluster_info[should_cluster] = []
                    current_new_cluster += 1
                else:
                    should_cluster = cluster_list[index]

                choose_action.append(index)
                cluster_info[should_cluster].append(mid)
                cluster_list.append(should_cluster)
                mid += 1

                link = index
                m1, m2 = rl['ids'][s + link]
                doc.link(m1, m2)

                path.append(index)

                #cluster_indexs = torch.cuda.LongTensor(cluster_info[should_cluster])
                #action_embedding_predict = torch.mean(action_embeddings[cluster_indexs],0,keepdim=True)
                #predict_action_embedding.append(action_embedding_predict)

            tmp_data.append(data)

            if rl["end"] == True:
                inside_index = 0
                manager_entropy = 0.0
                for data in tmp_data:
                    new_step = step
                    rl = data["rl"]
                    reward = doc.get_f1()

                    ave_reward.append(reward)
                    if len(ave_reward) >= MAX_AVE:
                        ave_reward = ave_reward[1:]
                    reward_log.log_value('reward', float(sum(ave_reward))/float(len(ave_reward)), new_step)  

                    scores_manager,representations_manager = get_score_representations(manager,data,dropout=nnargs["dropout_rate"])
                    doc_weight = (len(doc.mention_to_gold) + len(doc.mentions)) / 10.0 
                    baselines = []
                    for s,e in zip(rl["starts"],rl["ends"]):
                        score = score_softmax(torch.transpose(scores_manager[s:e],0,1))

                        ids = rl['ids'][s:e]
                        ana = ids[0, 1]
                        old_ant = doc.ana_to_ant[ana]
                        doc.unlink(ana)
                        costs = rl['costs'][s:e]
                        for ant_ind in range(e - s):
                            costs[ant_ind] = doc.link(ids[ant_ind, 0], ana, hypothetical=True, beta=1)
                        doc.link(old_ant, ana)
                        #costs -= costs.max()
                        #costs *= -doc_weight
                        costs = autograd.Variable(torch.from_numpy(costs).type(torch.cuda.FloatTensor))

                        if not score.size()[1] == costs.size()[0]:
                            #print "score.size not comparable with costs.size at",rl["did"]
                            continue
                        baseline = torch.sum(score*costs)
                        baselines.append(baseline)

                    optimizer_manager.zero_grad
                    manager_loss = None

                    for s,e,i in zip(rl["starts"],rl["ends"],range(len(rl["ends"]))):
                        baseline = baselines[i]
                        action = path[inside_index]
                        score = torch.squeeze(score_softmax(torch.transpose(scores_manager[s:e],0,1)))
                        this_cost = torch.log(score[action])*-1.0*(reward-baseline)
                        
                        if manager_loss is None:
                            manager_loss = this_cost
                        else:
                            manager_loss += this_cost
                        manager_entropy += torch.sum(score*torch.log(score+1e-7)).data.cpu().numpy()[0]
                        inside_index += 1

                    manager_loss.backward()
                    torch.nn.utils.clip_grad_norm(manager.parameters(), nnargs["clip"])
                    optimizer_manager.step()

                    ave_manager_entropy.append(manager_entropy)
                    if len(ave_manager_entropy) >= MAX_AVE:
                        ave_manager_entropy = ave_manager_entropy[1:]
                    entropy_log_manager.log_value('entropy', float(sum(ave_manager_entropy))/float(len(ave_manager_entropy)), new_step)

                    new_step += 1

                step = new_step
                tmp_data = []
                cluster_info = {0:[0]}
                cluster_list = [0]
                current_new_cluster = 1
                mid = 1
                predict_action_embedding = []
                choose_action = []
                path = []

        end_time = timeit.default_timer()
        print >> sys.stderr, "TRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr,"save model ..."
        #print "Top k",top_k
        print "Worker Hits",statistic["worker_hits"], "Manager Hits", statistic["manager_hits"], "Total",statistic["total"]
        print "Worker predict last", statistic["worker_predict_last"], "Manager predict last", statistic["manager_predict_last"]
        #torch.save(network_model, model_save_dir+"network_model_rl_worker.%d"%echo)
        #torch.save(ana_network, model_save_dir+"network_model_rl_manager.%d"%echo)
        
        print "DEV"
        #metric = performance.performance(dev_docs_iter,worker,manager) 
        metric = performance.performance(dev_docs_iter,manager) 
        print "Average:",metric["average"]
        print "TEST"
        metric = performance.performance(test_docs_iter,manager) 
        print "Average:",metric["average"]
        print
        sys.stdout.flush()

def get_score_representations(manager, data, dropout=0.0):
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

    output_manager, pair_score_manager, mention_pair_representations_manager = manager.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_spans,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout)
    ana_output_manager, ana_score_manager, ana_pair_representations_manager = manager.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature,dropout)

    mention_pair_representations_manager = autograd.Variable(torch.from_numpy(mention_pair_representations_manager).type(torch.cuda.FloatTensor))
    ana_pair_representations_manager = autograd.Variable(torch.from_numpy(ana_pair_representations_manager).type(torch.cuda.FloatTensor))
    reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

    scores_manager = torch.transpose(torch.cat((pair_score_manager,ana_score_manager),1),0,1)[reindex]
    representations_manager = torch.cat((mention_pair_representations_manager,ana_pair_representations_manager),0)[reindex]
    return scores_manager,representations_manager

if __name__ == "__main__":
    main()
