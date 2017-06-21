# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:37:38 2017

@author: Yuxian Meng, Peking University
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import json

file = open('./dicts_freq.json')
dicts_total_arc = json.load(open('./dicts.json'))
dicts = json.load(file)
pos_id, id2arc, arc2id = dicts_total_arc['pos_id'], dicts_total_arc['id2arc'], dicts_total_arc['arc2id']
word2id, id2word = dicts['word2id_freq'], dicts['id2word_freq']
un = "__UNKNOWN__" # unknown symbol
alpha = 0.25 # drop parameter


class Encoder(nn.Module): #TODO:compare LSTM/GRU; n layers; hidden_size, bi-directional, etc.
    """
    input: [POS_0, POS_1, ..., POS_l, word_0, ..., word_l]: 2*l, batch_size 
    output: l, batch, hidden_size
    
    """

    def __init__(self, pos_dim, hidden_size, nlayers = 1, npos = len(pos_id), 
                 rnn_type = 'GRU', dropout=0.1, bidirection = False, activate = nn.Tanh, 
                 nwords = len(word2id), words_dim = 100, fc_unit = 128):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder_pos = nn.Embedding(npos, pos_dim)
        self.encoder_word = nn.Embedding(nwords, words_dim)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(pos_dim+words_dim, hidden_size, nlayers, 
                              dropout=dropout, bidirectional = bidirection)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.bidirection = bidirection
        input_dim = self.hidden_size*2 if bidirection else self.hidden_size
        self.fc1 = nn.Linear(input_dim, fc_unit)
        self.fc2 = nn.Linear(input_dim, fc_unit)
        self.init_weights()
        self.activate = activate()
            
    def init_weights(self):
        initrange = 0.2; initrange2 = 0.05
        self.encoder_pos.weight.data.uniform_(-initrange, initrange)
        self.encoder_word.weight.data.uniform_(-initrange, initrange)
        init.xavier_normal(self.fc1.weight, gain=nn.init.calculate_gain('tanh')) #TODO：build a dict
        init.xavier_normal(self.fc2.weight, gain=nn.init.calculate_gain('tanh')) 
        self.fc1.bias.data.uniform_(-initrange2, initrange2)
        self.fc2.bias.data.uniform_(-initrange2, initrange2)
                          
                          
    def forward(self, input):
        seq_len, batch_size = input.size()[0], input.size()[1]
        l = seq_len //2
        input_pos, input_words = input[:l], input[l:]
        emb_pos = self.encoder_pos(input_pos) #l, batch_size, pos_dim
        emb_words = self.encoder_word(input_words) #l, batch_size, words_dim
        emb = torch.cat([emb_pos, emb_words], 2) #l, batch_size, words_dim+pos_dim
        emb_drop = emb    #TODO: compare with self.drop(emb)
        hidden_init = self.init_hidden(batch_size)
        output, hidden = self.rnn(emb_drop, hidden_init)# l, batch, hidden_size*2
        output_flat = output.view(l*batch_size, -1)
        features_1 = self.activate(self.fc1(output_flat)).view(l, batch_size,-1) # l, batch, fc_unit
        features_2 = self.activate(self.fc2(output_flat)).view(l, batch_size,-1) # l, batch, fc_unit
        features = []
        for i in range(l):
            for j in range(l):
                feature = features_1[i] + features_2[j] # batch, fc_unit
                features.append(feature)
        features = torch.stack(features, 0) # l^2, batch, fc_unit
        return output, features

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        num_direction = 2 if self.bidirection else 1
        dim1 = self.nlayers * num_direction
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(dim1, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(dim1, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(dim1, batch_size, self.hidden_size).zero_())

class Tanh3(nn.Module):
    def __init__(self):
       super(Tanh3, self).__init__()
       self.activate = nn.Tanh()
    def forward(self, x):
        return self.activate(x*x*x)


class Decoder(nn.Module): 
    """
    An MLP to decode arc scores from features extracted by Encoder.
    inputs: tensor of shape N*(N-1)/2, hidden_size of encoder
    outputs: scores of shape N*(N-1)/2, outokens(include none)    
    """
    def __init__(self, input_size, neuron_nums = [len(arc2id)*2-1],
                 activate = nn.Tanh, drop = 0.5,):
        super(Decoder, self).__init__()
        self.fcs = []
        self.nums = neuron_nums
        self.nums.insert(0, input_size)
        self.activate = activate()
        self.params = []
        for i in range(len(self.nums)-1):
            layer = nn.Linear(self.nums[i], self.nums[i+1])
            self.fcs.append(layer)
            self.params += [p for p in layer.parameters()]
        self.fcs = nn.ModuleList(self.fcs)
        self.params = nn.ParameterList(self.params) 
        self.init_weights()
        self.drop = drop
        self.dropout = nn.Dropout(self.drop)        
    
    def forward(self, input):
        output = input
        for layer in self.fcs:
            output = self.dropout(output)
            output = layer(output)
            output = self.activate(output)
        return output
             
    def init_weights(self):
        initrange = 0.05
        for layer in self.fcs:
            layer.bias.data.uniform_(-initrange, initrange)
            init.xavier_normal(layer.weight, gain=nn.init.calculate_gain('tanh'))

    
