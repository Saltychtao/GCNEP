import torch
from torch.utils.data import Dataset
from collections import defaultdict
import linecache
import os
import dill
import numpy as np


from utils.util import pad,load_pretrained
from dataloader.vocab import Vocab


def scatter(labels,max_len):
    array = [0 for i in range(max_len)]
    for idx in labels:
        array[idx] = 1
    return array


class EntityTypingDataset(Dataset):

    def __init__(self,filename,vocab,batch_size,share_vocab):

        self.share_vocab = share_vocab
        self.vocab = vocab
        self.read_file(filename,vocab)
        self.filename = filename
        self.batch_size = batch_size

    def read_file(self,filename,vocab):

        self.full_label_set = list(
            map(lambda x: x.replace('/', ' ').replace('_',' ').split(), list(sorted(vocab.ltoi, key=vocab.ltoi.get))))
        self.label_dict = defaultdict(lambda: [])
        self.label_set = set()
        self.length = 0
        cnt = 0
        with open(filename,'r') as f:
            for line in f:
                mention,labels,left_context,right_context,_= line.rstrip().split('\t')
                for label in labels.split():
                    self.label_dict[label].append(cnt)
                    self.label_set.add(label)
                if cnt % 1000 == 0:
                    print('\r{}'.format(cnt),end='')
                cnt += 1
                self.length += 1

    def process_line(self,line):
        mention,labels,left_context,right_context,_= line.rstrip().split('\t')
        example = dict()
        example['mention'] = mention.split()
        example['labels'] = labels.split()
        example['left_context'] = left_context.split()
        example['right_context'] = right_context.split()

        mention = [self.vocab.stoi[word] for word in example['mention']]
        left_context = [self.vocab.stoi[word] for word in example['left_context']]
        right_context = [self.vocab.stoi[word] for word in example['right_context']]

        instance = dict()
        instance['mention'] = mention
        instance['labels'] = labels.split()
        instance['left_context'] = left_context
        instance['right_context'] = right_context
        return instance

    def get_label_set(self):
        return self.label_set

    # FIX: remove the instances that contains the labels
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

    def __len__(self):
        return self.length

    def __getitem__(self,item):
        line = linecache.getline(self.filename,item + 1)
        instance = self.process_line(line)
        instance['labels_idx'] = scatter([self.vocab.ltoi[label] for label in instance['labels']],len(self.vocab.ltoi))
        if self.share_vocab:
            instance['full_labels'] = [[self.vocab.stoi[word] for word in label] for label in self.full_label_set]
        else:
            instance['full_labels'] = [[self.vocab.lwtoi[word] for word in label] for label in self.full_label_set]
        return instance

    @staticmethod
    def build_vocab(filenames,args):

        vocab = Vocab()
        vocab.stoi = {'<pad>':0,'<unk>':1}
        vocab.ltoi = {}
        vocab.lwtoi = {'<pad>':0,'<unk>':1}
        vocab.ftoi = {}
        for filepath in filenames:
            with open(filepath,'r') as f:
                for line in f:
                    mention,labels,left_context,right_context,_= line.rstrip().split('\t')
                    vocab.renew_vocab(mention.split(),'stoi')
                    vocab.renew_vocab(left_context.split(),'stoi')
                    vocab.renew_vocab(right_context.split(),'stoi')
                    for label in labels.split():
                        if not args.share_vocab:
                            vocab.renew_vocab(label.replace('/',' ').replace('_',' ').split(),'lwtoi')
                        else:
                            vocab.renew_vocab(label.replace('/',' ').replace('_',' ').split(),'stoi')
                    vocab.renew_vocab(labels.split(),'ltoi')

        return vocab

    @staticmethod
    def generate_dataset(args):
        filepaths = ['train.tsv', 'dev.tsv', 'test.tsv']
        for i in range(len(filepaths)):
            filepaths[i] = os.path.join(args.data_dir, filepaths[i])

        vocab = EntityTypingDataset.build_vocab(filepaths, args)
        torch.save(vocab, args.vocab_pth)

        print('Saved vocab')

        train_dataset = EntityTypingDataset(filepaths[0],vocab,args.batch_size,args.share_vocab)
        dev_dataset = EntityTypingDataset(filepaths[1],vocab,args.batch_size,args.share_vocab)
        test_dataset = EntityTypingDataset(filepaths[2], vocab, args.batch_size, args.share_vocab)
        print('\n Saving datasets.')
        torch.save(train_dataset, args.train_dataset_pth, pickle_module=dill)
        torch.save(dev_dataset,args.dev_dataset_pth,pickle_module=dill)
        torch.save(test_dataset, args.test_dataset_pth, pickle_module=dill)

    @staticmethod
    def generate_embedding(args,device):
        vocab = torch.load(args.vocab_pth)
        if args.word_pretrained_path is not None:
            if args.glove_pth is not None:
                args.word_pretrained = load_pretrained(args.glove_pth, vocab.stoi, dim=args.word_dim, device=device,pad_idx=args.padding_idx)
                torch.save(args.word_pretrained, args.word_pretrained_path)
            else:
                args.word_pretrained = torch.load(args.word_pretrained_path)

        if args.label_word_pretrained_path is not None and not args.share_vocab:
            if args.glove_pth is not None:
                args.label_word_pretrained = load_pretrained(args.glove_pth, vocab.lwtoi, dim=args.label_word_dim,device=device, pad_idx=args.padding_idx)
                torch.save(args.label_word_pretrained, args.label_word_pretrained_path)
            else:
                args.label_word_pretrained = torch.load(args.label_word_pretrained_path)

    @staticmethod
    def collate_fn(list_of_examples):
        batch = dict()
        mention = pad([x['mention'] for x in list_of_examples],0)
        left_context = pad([x['left_context'] for x in list_of_examples],0)
        right_context = pad([x['right_context'] for x in list_of_examples],0)
        labels = [x['labels_idx'] for x in list_of_examples]
        full_labels = pad(list_of_examples[0]['full_labels'],0)

        batch['mention'] = np.array(mention)
        batch['left_context'] = np.array(left_context)
        batch['right_context'] = np.array(right_context)
        batch['full_labels'] = np.array([full_labels for i in range(len(list_of_examples))])
        batch['labels'] = np.array(labels)

        return batch

    @staticmethod
    def load_dataset(args):
        vocab = torch.load(args.vocab_pth,pickle_module=dill)
        filepaths = ['train.tsv', 'dev.tsv', 'test.tsv']
        for i in range(len(filepaths)):
            filepaths[i] = os.path.join(args.data_dir, filepaths[i])
        train_dataset = EntityTypingDataset(filepaths[0],vocab,args.batch_size,args.share_vocab)
        dev_dataset = EntityTypingDataset(filepaths[1],vocab,args.batch_size,args.share_vocab)
        test_dataset = EntityTypingDataset(filepaths[2], vocab, args.batch_size, args.share_vocab)

        return train_dataset,dev_dataset,test_dataset

    @staticmethod
    def load_vocab(args):
        return torch.load(args.vocab_pth)
