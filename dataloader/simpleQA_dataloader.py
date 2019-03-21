import torch
from torch.utils.data import Dataset
from collections import defaultdict
import linecache
import os
import dill
import numpy as np
import random

from utils.util import pad,load_pretrained
from dataloader.vocab import SimpleQAVocab


class SimpleQADataset(Dataset):

    def __init__(self,filename,vocab,batch_size,ns=0,):

        self.vocab = vocab
        self.read_file(filename)
        self.filename = filename
        self.batch_size = batch_size
        self.ns = ns

    def read_file(self,filename):

        self.label_dict = defaultdict(lambda: [])
        self.label_set = set()
        cnt = 0
        self.length = 0
        with open(filename,'r') as f:
            for line in f:
                gold,neg,question = line.rstrip().split('\t')
                self.label_dict[int(gold)].append(cnt)
                self.label_set.add(int(gold) - 1)
                if cnt % 1000 == 0:
                    print('\r{}'.format(cnt),end='')
                cnt += 1
                self.length += 1
            print('\n')

    def process_line(self,line):
        gold,neg,question = line.rstrip().split('\t')

        question = [self.vocab.stoi.get(word,1) for word in question.split()]
        relations = []
        relations.append(int(gold)+1)
        for n in neg.split():
            try:
                idx = int(n) + 1
                relations.append(idx)
            except ValueError:
                pass
        return {
            'question': question,
            'relations': relations,
        }

    def get_label_set(self):
        return self.label_set

    def __len__(self):
        return self.length

    def __getitem__(self,item):
        line = linecache.getline(self.filename,item + 1)
        instance = self.process_line(line)
        if self.ns > 0:
            while len(instance['relations']) - 1 < self.ns:
                idx = random.randint(0,len(self.vocab.rtoi) - 1)
                if idx in instance['relations']:
                    continue
                instance['relations'].append(idx)
        else:
            while len(instance['relations']) - 1 < 200:
                instance['relations'].append(0)

        return instance

    @staticmethod
    def build_vocab(filenames,args):
        vocab = SimpleQAVocab()
        vocab.stoi = {'<pad>':0,'<unk>':1,'<relpad>':2}
        vocab.rtoi = {'<relpad>':0}
        with open(args.relation_file,'r') as f:
            for line in f.readlines():
                relation = line.rstrip()
                vocab.rtoi[relation] = len(vocab.rtoi)
                vocab.renew_vocab(relation.replace('.',' ').replace('_',' ').split(),'stoi')
        for filepath in filenames:
            with open(filepath,'r') as f:
                for line in f:
                    gold,neg,question = line.rstrip().split('\t')
                    vocab.renew_vocab(question.split(),'stoi')

        for rel,idx in vocab.rtoi.items():
            relation_words = rel.replace('.',' ').replace('_',' ').split()
            vocab.relIdx2wordIdx[idx] = [vocab.stoi[w] for w in relation_words]

        return vocab

    @staticmethod
    def generate_dataset(args):
        filepaths = ['train.tsv','dev.tsv','test.tsv']
        for i in range(len(filepaths)):
            filepaths[i] = os.path.join(args.data_dir,filepaths[i])

        vocab = SimpleQADataset.build_vocab(filepaths,args)
        torch.save(vocab,args.vocab_pth)

        print('Saved Vocab')

    @staticmethod
    def generate_embedding(args,device):
        vocab = torch.load(args.vocab_pth)
        args.word_pretrained = load_pretrained(args.glove_pth,vocab.stoi,dim=args.word_dim,device=device,pad_idx=args.padding_idx)
        torch.save(args.word_pretrained,args.word_pretrained_pth)
        # args.label_pretrained = load_pretrained(args.transE_pth,vocab.rtoi,dim=args.relation_dim,device=device)
        # torch.save(args.label_pretrained,args.label_pretrained_pth)

    @staticmethod
    def collate_fn(list_of_examples):
        question = np.array(pad([x['question'] for x in list_of_examples],0))
        relation = [x['relations'] for x in list_of_examples]

        labels = [0] * len(relation)

        return {
            'question':question,
            'relation':np.array(relation),
            'labels':np.array(labels)
        }

    @staticmethod
    def load_dataset(args):
        vocab = torch.load(args.vocab_pth,pickle_module=dill)
        filepaths = ['train.tsv','dev.tsv','test.tsv']
        for i in range(len(filepaths)):
            filepaths[i] = os.path.join(args.data_dir,filepaths[i])

        train_dataset = SimpleQADataset(filepaths[0],vocab,args.batch_size,args.ns)
        dev_dataset = SimpleQADataset(filepaths[1],vocab,args.batch_size,ns=0)
        test_dataset = SimpleQADataset(filepaths[2],vocab,args.batch_size,ns=0)

        return train_dataset, dev_dataset, test_dataset

    @staticmethod
    def load_vocab(args):
        return torch.load(args.vocab_pth)
