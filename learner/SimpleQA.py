import os
import random
from overrides import overrides
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


import torch
import torch.nn as nn

from utils.module import LSTMEncoder,mean_pool,max_pool,GateNetwork
# from propagator.GCN import RGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleQA(nn.Module):

    def __init__(self,args):
        super(SimpleQA, self).__init__()

        if args.word_pretrained is None:
            self.word_embedding = nn.Embedding(args.n_words,args.word_dim,args.padding_idx)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(args.word_pretrained,freeze=args.freeze)

        self.relation_embedding = nn.Embedding(args.n_relations,args.relation_dim,args.padding_idx)

        self.word_encoder = LSTMEncoder(
            input_size=args.word_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True
        )

        self.question_encoder = LSTMEncoder(
            input_size=2*args.hidden_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True
        )

        self.gate = GateNetwork(2*args.hidden_dim)

        self.loss_fn = nn.MultiMarginLoss(margin=args.margin)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=args.lr)

        self.ns = args.ns
        self.score_function = nn.CosineSimilarity(dim=2)

        self.all_relation_words = args.all_relation_words
        self.all_relation_names = args.all_relation_names

        self.n_relations = args.n_relations
        self.args = args
        self.dropout = nn.Dropout(p=0.5)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self,question,relation):

        n_rels = relation.size()[1]
        question_length = (question != self.args.padding_idx).sum(dim=1).long().to(device)
        question = self.word_embedding(question)
        # question = self.dropout(question)
        low_question_repre = self.word_encoder(question,question_length,need_sort=True)[0]

        high_question_repre = self.question_encoder(low_question_repre,question_length,need_sort=True)[0]
        question_repre = (low_question_repre + high_question_repre).mean(1)  # bsize * hidden

        # all_relations = torch.tensor([i for i in range(self.n_relations)]).to(device)
        all_relation_words = torch.tensor(self.all_relation_words).to(device)  # n_relations * max_len

        # single_relation_repre = self.relation_embedding(all_relations)
        all_relation_names = torch.tensor(self.all_relation_names).to(device)
        relation_names_lengths = (all_relation_names != self.args.padding_idx).sum(dim=-1).long().to(device)
        
        all_relation_names = self.word_embedding(all_relation_names)
        
        relation_names_repre = self.word_encoder(all_relation_names,relation_names_lengths,need_sort=True)[0]

        relation_words_lengths = (all_relation_words != self.args.padding_idx).sum(dim=-1).long().to(device)
        relation_words_repre = self.word_embedding(all_relation_words)
        relation_words_repre = self.dropout(relation_words_repre)
        relation_words_repre = self.word_encoder(relation_words_repre,relation_words_lengths,need_sort=True)[0]
        relation_repre = torch.cat([relation_names_repre,relation_words_repre],dim=1).mean(1)

        # relation_repre = self.gate(relation_names_repre,relation_words_repre)

        relation_repre = relation_repre[relation,:]  # bsize * n_rels * hidden

        scores = self.score_function(relation_repre,question_repre.unsqueeze(1).repeat(1,n_rels,1))

        return scores

    def train_epoch(self,train_iter):
        self.train()

        total_batch = len(train_iter)
        total = 0.
        loss = 0.0
        cur_batch = 1
        correct = 0
        for batch in train_iter:
            question = torch.tensor(batch['question']).to(device)
            relation = torch.tensor(batch['relation']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            bsize = question.size()[0]

            scores = self.forward(question,relation)  # bsize * (1 + ns)
            batch_loss = self.loss_fn(scores,labels)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            cur_batch += 1

            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += bsize

            loss += batch_loss
            print('\r Batch {}/{}, Training Loss:{:.4f}, Training Acc:{:.2f}'.format(cur_batch,total_batch,loss/cur_batch,correct/total*100),end='')

    def evaluate(self,dev_iter):
        self.eval()
        correct = 0.
        total = 0.
        for batch in dev_iter:
            question = torch.tensor(batch['question']).to(device)
            relation = torch.tensor(batch['relation']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            bsize = question.size()[0]

            relation_mask = (2*(relation != 0) -1).float() # 1 -> 1, 0 -> -1
            scores = self.forward(question,relation) # bsize * (1 + neg_num)
            correct += ((scores*relation_mask).argmax(dim=1) == labels).sum().item()
            total += bsize

        return correct / total
