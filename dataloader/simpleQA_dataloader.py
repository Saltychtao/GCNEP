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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pairwise_distances(x,y=None):

    x_norm = (x**2).sum(1).view(-1,1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1,-1)
    else:
        y=x
        y_norm = x_norm.view(1,-1)

    dist = x_norm + y_norm - 2.0*torch.mm(x,torch.transpose(y,0,1))
    return dist


class SimpleQADataset(Dataset):

    def __init__(self,filename,vocab,batch_size,ns=0,train=True):

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
                self.label_set.add(int(gold))
                cnt += 1
                self.length += 1

    def process_line(self,line):
        gold,neg,question = line.rstrip().split('\t')

        question = [self.vocab.stoi.get(word,1) for word in question.split()]
        relations = []
        relations.append(int(gold))
        for n in neg.split():
            try:
                idx = int(n)
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

    def get_subset(self,labels,mode):
        idxs = []
        if mode == 'seen':
            for label in labels:
                idxs.extend(self.label_dict[label])

        elif mode == 'unseen':
            excluded = []
            for label in labels:
                excluded.extend(self.label_dict[label])
            idxs = set([i for i in range(self.__len__())]) - set(excluded)
        return torch.utils.data.Subset(self,list(set(idxs)))

    def get_raw_instance(self,idx):
        return linecache.getline(self.filename,idx+1)

    def __getitem__(self,item):
        line = linecache.getline(self.filename,item + 1)
        instance = self.process_line(line)
        pos = instance['relations'][0]
        if self.ns > 0:
            while len(instance['relations']) - 1 < self.ns:
                idx = random.randint(1,len(self.vocab.rtoi) - 1)
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
        vocab.rtoi = {}
        vocab.itor = {}
        with open(args.relation_file,'r') as f:
            for line in f.readlines():
                relation = line.rstrip()
                if relation not in vocab.rtoi:
                    vocab.itor[len(vocab.rtoi)] = relation
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
    def generate_vocab(args):
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

    @staticmethod
    def generate_relation_embedding(args,device):
        vocab = torch.load(args.vocab_pth)
        vecs = np.random.normal(0,1,(len(vocab.rtoi),50))
        vecs[args.padding_idx] = np.zeros((50))
        cnt = 0
        with open(args.relation_vec_pth,'r') as f:
            for line in f:
                splited = line.split('\t')
                word = splited[0]
                vec = splited[1]
                if word not in vocab.rtoi:
                    continue
                cnt += 1
                vec = [float(f) for f in vec.split()]
                vecs[vocab.rtoi[word]] = vec
            print('Found word vectors: {}/{}'.format(cnt,len(vocab.rtoi)))
        relation_pretrained = torch.from_numpy(vecs).float().to(device)
        torch.save(relation_pretrained,args.relation_pretrained_pth)

    @staticmethod
    def generate_graph(args,device):
        vocab = torch.load(args.vocab_pth)
        print('Total Relations: {}'.format(len(vocab.rtoi)))
        relation_pretrained = torch.load(args.relation_pretrained_pth)
        distance_matrix = pairwise_distances(relation_pretrained)
        print(distance_matrix.mean())
        adj_matrix = (distance_matrix < args.threshold).long().cpu().numpy()
        print(adj_matrix.sum())

        torch.save(adj_matrix,args.relation_adj_matrix_pth)

    @staticmethod
    def collate_fn(list_of_examples):
        question = np.array(pad([x['question'] for x in list_of_examples],0))

        relation = [x['relations'] for x in list_of_examples]

        labels = [0] * len(relation)

        return {
            'question':question,
            'relation':np.array(relation),
            'labels':np.array(labels),
        }

    @staticmethod
    def load_dataset(train_fname,dev_fname,test_fname,vocab_pth,args):
        vocab = torch.load(vocab_pth,pickle_module=dill)

        train_dataset = SimpleQADataset(train_fname,vocab,args.batch_size,args.ns)
        dev_dataset = SimpleQADataset(dev_fname,vocab,args.batch_size,ns=0)
        test_dataset = SimpleQADataset(test_fname,vocab,args.batch_size,ns=0)

        return train_dataset, dev_dataset, test_dataset

    @staticmethod
    def load_vocab(args):
        return torch.load(args.vocab_pth)

    @staticmethod
    def load_graph(args):
        return torch.load(args.graph_pth)
