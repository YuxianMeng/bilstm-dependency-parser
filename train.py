# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:28:31 2017

@author: Yuxian Meng, Peking University
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model import Encoder, Decoder, Tanh3
from data_input import batch_input
import torch
from torch.autograd import Variable
from six.moves import xrange
import time
from graph2tree import graph2tree
import copy
import json
from decoder import DependencyDecoder
mst = DependencyDecoder()

file = open('./dicts_freq.json')
dicts_total_arc = json.load(open('./dicts.json'))
dicts = json.load(file)
pos_id, id2arc, arc2id = dicts_total_arc['pos_id'], dicts_total_arc['id2arc'], dicts_total_arc['arc2id']
word2id, id2word = dicts['word2id_freq'], dicts['id2word_freq']
un = "__UNKNOWN__" # unknown symbol
alpha = 0.25 # drop parameter
def stats_scores(dic_predicts, dic_trues):
    """
    input: 2 lists contains dicts like {(i,j):k, ...}, len(list) = batch_size
    return: accuracies and recalls
    """
    total_trues = 0
    total_predicts = 0
    true_positive_label = 0
    true_positive_arc = 0
    for i in range(len(dic_predicts)):
        for key in dic_predicts[i]:
            if key in dic_trues[i]:
                true_positive_arc += 1
                if dic_predicts[i][key] == dic_trues[i][key]:
                    true_positive_label += 1
        total_trues += len(dic_trues[i])
        total_predicts += len(dic_predicts[i])
        
    accuracy_label = true_positive_label / (total_predicts+1)
    accuracy_arc = true_positive_arc / (total_predicts+1)  #+1 to smooth
    recall_label = true_positive_label / (total_trues+1)
    recall_arc = true_positive_arc / (total_trues+1)
    return np.array([accuracy_label, accuracy_arc, recall_label, recall_arc])
 
def train_model(encoder, decoder1, decoder2, decoder3,optimizer1, optimizer2, optimizer3, optimizer4,
                num_epochs=5, use_gpu = False, batch_size = 1, clip = 5, proj = True):
    since = time.time()
    best_model = [encoder, decoder1, decoder2, decoder3]
    print_freq = int(200/batch_size)
    eval_freq = 3
    best_acc = 0
    if use_gpu:
        encoder.cuda()
        decoder1.cuda()
        decoder2.cuda()
        decoder3.cuda()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                encoder.train(True)  # Set model to training mode --enable dropout
                decoder1.train(True)
                decoder2.train(True)
                decoder3.train(True)
                train = True
            else:
                if (epoch+1) % eval_freq != 0: #  evaluate every eval_freq epochs
                    continue
                encoder.train(False)  # Set model to evaluate mode
                decoder1.train(False)
                decoder2.train(False)
                decoder3.train(False)
                train = False
            steps = data_loader.ls[0]//batch_size if train else data_loader.ls[1]//batch_size  
            running_loss = 0.0
            running_stats = np.array([0.0,0.0,0.0,0.0])
            # Iterate over data.
            for batch in xrange(steps):
                loss_arc = 0.0
                loss_label = Variable(torch.zeros(1)).cuda()
                # zero the parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                optimizer4.zero_grad()
                # get the inputs
                batch_data = data_loader.next_batch(batch_size = batch_size, 
                                                    train = train, drop = train) #TODO:disnable drop
                arcs_predicts = [] #list to store arcs_predict for each item in batch
                arcs_trues = [] #先只对生成的树计算F-score
                for item in range(batch_size):
                    loss_arc_single = Variable(torch.zeros(1)).cuda()
                    data = batch_data[item]                   
                    input_seqs, arcs_dic, puncs, roots = data
                    
                    input_seqs = np.reshape(input_seqs, (input_seqs.shape[0],1))  
                    if use_gpu:
                        inputs = Variable(torch.from_numpy(input_seqs).cuda())
                    else:
                        inputs = Variable(torch.from_numpy(input_seqs))
                    # forward
                    output, features = encoder(inputs) #seq_len, batch, hidden_size; l*l, batch, fc_unit
                    seq_len = output.size(0) 
                    arcs, root = graph2tree(seq_len, arcs_dic, roots) # graph2tree
                    arcs_trues.append(arcs)
                    gold = np.array([-1 for _ in range(seq_len+1)]) #golden heads
                    for arc in arcs:
                        h, m = arc
                        gold[m+1] = h+1
                    gold[root+1] = 0
#                    print(features.size())
                    features = features[:, 0, :] #设batchsize=1, l*l, fc_unit
#                    features_root = output[:, 0, :] # N, 1, hidden
                    scores_arc = decoder1(features) # N*N, 1
#                    scores_root = decoder2(features_root).view(1, seq_len) #1, N #如果没有root呢？
                    scores_root = Variable(torch.zeros(1, seq_len))
                    scores_arc = scores_arc.view(seq_len, seq_len) # N, N
                    scores_pad = Variable(-torch.zeros(seq_len+1, 1)) #N+1, 1
                    if use_gpu:
                        scores_root = scores_root.cuda()
                        scores_pad = scores_pad.cuda()
                    # use MST to compute predicted arcs and loss according to scores
                    scores_total = torch.cat([scores_root, scores_arc], 0) #N+1, N
                    scores_total = torch.cat([scores_pad, scores_total], 1) # N+1, N+1
                    score_numpy = scores_total.cpu().data.numpy()
#                    print(score_numpy)
#                    print(scores_total)
#                    break
                    heads_train, _ = mst.parse_proj(score_numpy, gold, hinge) # used to compute margin loss
                    heads_predict, score_predict = mst.parse_proj(score_numpy, None,) # true predict
#                    print(gold, heads)
#                    print(sum([1 if gold[i] == heads[i] else 0 for i in range(len(gold))])/len(gold))    
                    label_features = [] #train decoder_label
                    target = []
                    for m, h in enumerate(gold[1:]): #label loss
                        if h != 0:
                            h -= 1    
                            feature = features[m*seq_len+h].view(1,-1) # 1, hidden
                            label_features.append(feature)
                            target.append(arcs[(h,m)])
                    label_features = torch.cat(label_features, 0) # N, hidden
                    target = Variable(torch.from_numpy(np.array(target))).cuda() #N
                    label_scores = decoder3(label_features) # N * label_types
                    loss_label += torch.nn.functional.cross_entropy(label_scores, target)
                    arcs_predict = {}   #暂时只算arc，不用算label，设为1即可
        
                    for m, h in enumerate(heads_predict[1:]): #predict
                        if h != 0:
                            h -= 1
#                            arcs_predict[(h,m)] = 1
                            feature = features[m*seq_len+h].view(1,-1) # 1, hidden
                            label_scores = decoder3(feature).view(-1) # label_types]
                            label_score, label_predict = label_scores.max(0)
                            label_predict = int(label_predict.cpu().data[0])
                            if  label_predict > len(arc2id)-1:
                                label_predict -= (len(arc2id)-1)
                                h,m = m, h
                            if label_predict and h not in puncs and m not in puncs:                                                     
                                arcs_predict[(h, m)] = label_predict
                    diff = 0
                    scores_gold = 0
                    for i, (h, g) in enumerate(zip(heads_train, gold)):
                        scores_gold += scores_total[g][i]
                        if h != g : 
#                            if (g-1, i-1) in arcs and not arcs[(g-1, i-1)]:
#                                continue  #若arc的label为None则不参与训练
                            diff +=1
                            loss_arc_single += scores_total[h][i] - scores_total[g][i]
                    
                    margin = delta * diff
                    if diff and loss_arc_single.data[0] + margin > 0:
                        loss_arc += loss_arc_single + margin  
                    arcs_predicts.append(arcs_predict)
                    if batch % (print_freq*1) ==0 and item == 0:
                        print(heads_predict)
                        print(gold)
                        print(score_predict, scores_gold.data[0])
#                        print(score_numpy[1])
#                        print(scores_total.data[1])
#                        print(arcs_predict)
#                        print(arcs)

                # statistics
                loss_arc /= batch_size
                loss_label /= (batch_size*20)
                loss = loss_arc + loss_label
#                arcs_trues = [item[1] for item in batch_data]
#                print(len(arcs_trues), len(arcs_predicts))
#                stats = non_zero_recall(arcs_predicts, arcs_trues)
                stats = stats_scores(arcs_predicts, arcs_trues)
#                print(stats)
                running_loss += loss.data[0]     
                running_stats += stats
                f_score_label = 2*stats[0]*stats[2]/(stats[0]+stats[2]+0.001)
                f_score_arc = 2*stats[1]*stats[3]/(stats[1]+stats[3]+0.001) #smooth                
                
                # backward + optimize only if in training phase
                if phase == 'train' and loss.data[0] > 0: #loss == 0就不用BP了
#                    print(loss.data[0])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip) #grad_clip
                    torch.nn.utils.clip_grad_norm(decoder1.parameters(), clip) #grad_clip
                    torch.nn.utils.clip_grad_norm(decoder2.parameters(), clip)                             
                    torch.nn.utils.clip_grad_norm(decoder3.parameters(), clip)
                    optimizer1.step() 
                    optimizer2.step()
                    optimizer3.step()
                    optimizer4.step()
                    if batch % print_freq == 0:
                        print('batch:{}, loss:{:4f} = arc:{:4f}+ label:{:4f}, f-label:{:4f}, f-arc:{:4f}'. \
                              format(batch, loss.data[0],loss_arc.data[0], loss_label.data[0], f_score_label, f_score_arc))

            epoch_loss = running_loss / steps
            epoch_stats = running_stats / steps
            epoch_label_fscore = 2*epoch_stats[0]*epoch_stats[2]/(epoch_stats[0]+epoch_stats[2]+0.001)
            epoch_arc_fscore = 2*epoch_stats[1]*epoch_stats[3]/(epoch_stats[1]+epoch_stats[3]+0.001)
            print('{} Loss: {:.4f}, fscore:{}, {}'.format(phase, 
                  epoch_loss, epoch_label_fscore, epoch_arc_fscore))

            # deep copy the model
            if phase == 'val' and epoch_label_fscore > best_acc:
                best_acc = epoch_label_fscore
                models = [encoder, decoder1, decoder2, decoder3]
                best_model = copy.deepcopy(models)
                
        if (epoch+1) % 5 == 0: #save best model internally
            models = [encoder, decoder1, decoder2, decoder3]
            for i in range(len(models)):
                torch.save(models[i].state_dict(), save_to[i])
            print("model saved")                

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val precision: {:4f}'.format(best_acc))

    return best_model      

if __name__ == "__main__": #TODO:try tanh3, different drop/ hiddensize
    #参数
    data_dirs = ['./data_train', './data_eval']; use_gpu = True; batch_size = 16
    hidden_size = 128; clip = 5; weight_decay = 1e-4; fc_unit = 128; delta = 0.05
    hinge = 0.05
    encoder = Encoder(pos_dim = 64, words_dim = 64, hidden_size = hidden_size, 
                      bidirection = True, nlayers = 2, rnn_type = 'LSTM', dropout=0.1,
                      fc_unit = fc_unit, activate = torch.nn.ELU)
    input_dim = hidden_size * 4 if encoder.bidirection else hidden_size * 2
    decoder_arc = Decoder(input_size = fc_unit, drop=0.1, #TODO:activate = Tanh3
                      neuron_nums = [1], activate = torch.nn.ELU)
    decoder_root = Decoder(input_size = input_dim//2, drop=0.1,
                      neuron_nums = [fc_unit, 1], activate = torch.nn.ELU)
    decoder_label = Decoder(input_size = fc_unit, drop= 0.1,
                      neuron_nums = [128, 2*len(arc2id)-1],) #activate = nn.Tanh)
    description = 'hinge_{}_delta_{}_ELU'.format(hinge,delta)
    description_pretrained = 'hinge_{}_delta_{}_ELU'.format(hinge,delta)
    pretrained = ['./encoder_{}.pth'.format(description_pretrained, ), 
               './decoder1_{}.pth'.format(description_pretrained, ), 
               './decoder2_{}.pth'.format(description_pretrained, ), 
               './decoder3_{}.pth'.format(description_pretrained, )]
    save_to = ['./encoder_{}.pth'.format(description, ), 
               './decoder1_{}.pth'.format(description, ), 
               './decoder2_{}.pth'.format(description, ), 
               './decoder3_{}.pth'.format(description, )]
#    #load saved models
#    encoder.load_state_dict(torch.load(pretrained[0]))
#    decoder_arc.load_state_dict(torch.load(pretrained[1]))
#    decoder_root.load_state_dict(torch.load(pretrained[2]))
#    decoder_label.load_state_dict(torch.load(pretrained[3]))
#    
    #RMSprop
    optimizer1 = torch.optim.RMSprop(encoder.parameters(), lr = 1e-3, weight_decay = weight_decay)
    optimizer2 = torch.optim.RMSprop(decoder_arc.parameters(), lr = 5e-4, weight_decay = weight_decay)
    optimizer3 = torch.optim.RMSprop(decoder_root.parameters(), lr = 5e-4, weight_decay = weight_decay)
    optimizer4 = torch.optim.RMSprop(decoder_label.parameters(), lr = 5e-4)

    data_loader = batch_input(*data_dirs)
    encoder_best, decoder1, decoder2, decoder3 = train_model(encoder, decoder_arc, decoder_root,
                                             decoder_label, optimizer1, optimizer2, optimizer3, 
                                             optimizer4, num_epochs = 50,
                                             use_gpu = use_gpu, clip = clip,
                                             batch_size = batch_size)
#    encoder_best, decoder_best = encoder, decoder
    torch.save(encoder_best.state_dict(), save_to[0])
    torch.save(decoder1.state_dict(), save_to[1])
    torch.save(decoder2.state_dict(), save_to[2])
    torch.save(decoder3.state_dict(), save_to[3])
    




