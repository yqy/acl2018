#coding=utf8

import sys
import os
import json
import random
import numpy
import numpy as np
import timeit

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
import performance

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, "PID", os.getpid()

torch.cuda.set_device(args.gpu)
 
def main():

    DIR = args.DIR
    embedding_file = args.embedding_dir

    embedding_matrix = numpy.load(embedding_file)
    "Building torch model"
    network_model = network.Network(nnargs["pair_feature_dimention"],nnargs["mention_feature_dimention"],nnargs["word_embedding_dimention"],nnargs["span_dimention"],1000,nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix).cuda()

    reduced=""
    if args.reduced == 1:
        reduced="_reduced"

    print >> sys.stderr,"prepare data for train ..."
    train_docs = DataReader.DataGnerater("train"+reduced)
    print >> sys.stderr,"prepare data for dev and test ..."
    dev_docs = DataReader.DataGnerater("dev"+reduced)
    test_docs = DataReader.DataGnerater("test"+reduced)

    l2_lambda = 1e-6
    lr = nnargs["lr"]
    dropout_rate = nnargs["dropout_rate"]
    epoch = nnargs["epoch"]

    model_save_dir = "./model/"
   
    last_cost = 0.0
    all_best_results = {
        'thresh': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
        }
  
    optimizer = optim.RMSprop(network_model.parameters(), lr=lr, eps = 1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

    for echo in range(epoch):

        scheduler.step()

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        pair_cost_this_turn = 0.0
        ana_cost_this_turn = 0.0

        pair_nums = 0
        ana_nums = 0

        for data in train_docs.train_generater(shuffle=True):
            
            mention_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
            mention_span = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
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

            gold = target.tolist()
            anaphoricity_gold = anaphoricity_target.tolist()

            pair_nums += len(gold)
            ana_nums += len(anaphoricity_gold)

            lable = autograd.Variable(torch.cuda.FloatTensor([gold]))
            ana_lable = autograd.Variable(torch.cuda.FloatTensor([anaphoricity_gold]))

            output,_,_ = network_model.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)
            ana_output,_,_ = network_model.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature, dropout_rate)

            optimizer.zero_grad()

            loss = F.binary_cross_entropy(output,lable,size_average=False)/train_docs.scale_factor
            ana_loss = F.binary_cross_entropy(ana_output,ana_lable,size_average=False)/train_docs.anaphoricity_scale_factor

            pair_cost_this_turn += loss.data[0]*train_docs.scale_factor
            ana_cost_this_turn += ana_loss.data[0]*train_docs.anaphoricity_scale_factor

            loss_all = loss + ana_loss
            loss_all.backward()
            optimizer.step()

        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTrain epoch",echo,"Pair total cost:",pair_cost_this_turn/float(pair_nums),"Anaphoricity total cost", ana_cost_this_turn/float(ana_nums)
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr, "Learning Rate",lr

        gold = []
        predict = []

        ana_gold = []
        ana_predict = []

        for data in dev_docs.train_generater(shuffle=False):
            
            mention_index = autograd.Variable(torch.from_numpy(data["mention_word_index"]).type(torch.cuda.LongTensor))
            mention_span = autograd.Variable(torch.from_numpy(data["mention_span"]).type(torch.cuda.FloatTensor))
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

            gold += target.tolist()
            ana_gold += anaphoricity_target.tolist()

            output,_,_ = network_model.forward_all_pair(nnargs["word_embedding_dimention"],mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,0.0)
            predict += output.data.cpu().numpy()[0].tolist()

            ana_output,_,_ = network_model.forward_anaphoricity(nnargs["word_embedding_dimention"], anaphoricity_index, anaphoricity_span, anaphoricity_feature, 0.0)
            ana_predict += ana_output.data.cpu().numpy()[0].tolist()
        
        gold = numpy.array(gold,dtype=numpy.int32)
        predict = numpy.array(predict)

        best_results = {
            'thresh': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

        thresh_list = [0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
        for thresh in thresh_list:
            evaluation_results = get_metrics(gold, predict, thresh)
            if evaluation_results["f1"] >= best_results["f1"]:
                best_results = evaluation_results
 
        print "Pair accuracy: %f and Fscore: %f with thresh: %f"\
                %(best_results["accuracy"],best_results["f1"],best_results["thresh"])
        sys.stdout.flush() 

        if best_results["f1"] >= all_best_results["f1"]:
            all_best_results = best_results
            print >> sys.stderr, "New High Result, Save Model"
            torch.save(network_model, model_save_dir+"network_model_pretrain.best")

        ana_gold = numpy.array(ana_gold,dtype=numpy.int32)
        ana_predict = numpy.array(ana_predict)
        best_results = {
            'thresh': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        for thresh in thresh_list:
            evaluation_results = get_metrics(ana_gold, ana_predict, thresh)
            if evaluation_results["f1"] >= best_results["f1"]:
                best_results = evaluation_results
        print "Anaphoricity accuracy: %f and Fscore: %f with thresh: %f"\
                %(best_results["accuracy"],best_results["f1"],best_results["thresh"])
        sys.stdout.flush() 

        if (echo+1)%10 == 0:
            best_network_model = torch.load(model_save_dir+"network_model_pretrain.best") 
            print "DEV:"
            performance.performance(dev_docs,best_network_model)
            print "TEST:"
            performance.performance(test_docs,best_network_model)

    ## output best
    print "In sum, anaphoricity accuracy: %f and Fscore: %f with thresh: %f"\
        %(best_results["accuracy"],best_results["f1"],best_results["thresh"])
    sys.stdout.flush()

def get_metrics(gold, predict, thresh):
    pred = np.clip(np.floor(predict / thresh), 0, 1)
    p, r = (0, 0) if pred.sum() == 0 else \
    (precision_score(gold, pred), recall_score(gold, pred))
    return {
        'thresh': thresh,
        'accuracy': average_precision_score(gold, predict),
        'precision': p,
        'recall': r,
        'f1': 0 if p == 0 or r == 0 else 2 * p * r / (p + r)
    } 

if __name__ == "__main__":
    main()
