import sys
sys.path.append('..')

from torchtext import data
import os
import torch
import torch.nn as nn
import dill

import numpy as np
from lib.module import LSTMEncoder
from lib.utils.util import load_pretrained
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.nn.util import add_positional_features,weighted_sum

from evaluate  import strict,loose_macro,loose_micro
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def one_hot_to_labels(tensor,one_label=None):
    # tensor bsize * N_C
    bsize,_ = tensor.size()
    if one_label is not None:
        labels = [[label.item()] for label in one_label]
    else:
        labels = [[] for i in range(bsize)]
    nonzeros = tensor.nonzero()
    for idx_label in nonzeros:
        idx = idx_label[0]
        label = idx_label[1].item()
        labels[idx].append(label)
    return labels

def scatter(labels,max_len):
    array = [0 for i in range(max_len)]
    for idx in labels:
        array[idx] = 1
    return array



class EntityTypingDataset(Dataset):

    def __init__(self,filename,vocab,batch_size):

        self.dataset = self.read_file(filename,vocab)
        self.batch_size = batch_size
        self.vocab = vocab

    def read_file(self,filename,vocab):
        dataset = []
        self.label_set = list(map(lambda x: x.replace('/',' ').split(),list(sorted(vocab.ltoi,key=vocab.ltoi.get))))
        cnt = 0
        with open(filename,'r') as f:
            for line in f:
                mention,labels,left_context,right_context= line.rstrip().split('\t')
                example = dict()
                example['mention'] = mention.split()
                example['labels'] = labels.split()
                example['left_context'] = left_context.split()
                example['right_context'] = right_context.split()

                mention = [vocab.stoi[word] for word in example['mention']]
                left_context = [vocab.stoi[word] for word in example['left_context']]
                right_context = [vocab.stoi[word] for word in example['right_context']]


                instance = dict()
                instance['mention'] = mention
                instance['labels'] = labels.split()
                instance['left_context'] = left_context
                instance['right_context'] = right_context

                dataset.append(instance)
                if cnt % 1000 == 0:
                    print('\r{}'.format(cnt),end='')
                cnt += 1

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,item):
        instance = self.dataset[item]
        instance['labels'] = scatter([self.vocab.ltoi[label] for label in instance['labels']], len(self.vocab.ltoi))
        instance['full_labels'] = [[self.vocab.lwtoi[word] for word in label ] for label in self.label_set]
        return instance


def pad(data,pad_idx,max_len=None):
    if max_len is None:
        max_len = max([len(instance) for instance in data])
    return [instance + [pad_idx]*max((max_len-len(instance),0)) for instance in data]


def collate_fn(list_of_examples):
    batch = dict()
    mention = pad([x['mention'] for x in list_of_examples],0)
    left_context = pad([x['left_context'] for x in list_of_examples],0)
    right_context = pad([x['right_context'] for x in list_of_examples],0)
    labels = [x['labels'] for x in list_of_examples]
    full_labels = pad(list_of_examples[0]['full_labels'],0)

    batch['mention'] = mention
    batch['left_context'] = left_context
    batch['right_context'] = right_context
    batch['full_labels'] = [full_labels for i in range(len(list_of_examples))]
    batch['labels'] = labels

    return batch


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, args):
        "docstring"
        super(SelfAttentiveEncoder, self).__init__()

        self.lstm = LSTMEncoder(
            input_size=args.word_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True
        )

        self.proj = nn.Sequential(
            nn.Linear(2*args.hidden_dim,args.attention_dim),
            nn.Tanh(),
            nn.Linear(args.attention_dim,1),
        )

    def forward(self,c_l,c_l_lengths,c_r,c_r_lengths):

        c_l_repre,_ = self.lstm(c_l,c_l_lengths,need_sort=True)
        c_r_repre,_ = self.lstm(c_r,c_r_lengths,need_sort=True)

        matrix = torch.cat([c_l_repre,c_r_repre],dim=1)

        l_proj = self.proj(c_l_repre)  # bsize * (length_l) * 1
        r_proj = self.proj(c_r_repre)

        weights = torch.cat([l_proj,r_proj],dim=1).squeeze()
        weights = torch.nn.functional.softmax(weights,dim=1)

        return weighted_sum(matrix,weights)


class Model(nn.Module):

    def __init__(self, args):
        "docstring"
        super(Model, self).__init__()

        if args.word_pretrained is None:
            self.word_embedding = nn.Embedding(args.n_words,args.word_dim,args.padding_idx)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(args.word_pretrained,freeze=False)


        self.label_embedding = nn.Embedding(args.n_labels,args.label_dim)

        if args.label_word_pretrained is None:
            self.label_word_embedding = nn.Embedding(args.n_labelwords,args.label_word_dim,args.padding_idx)
        else:
            self.label_word_embedding = nn.Embedding.from_pretrained(args.label_word_pretrained,freeze=False)

        self.mention_encoder = BagOfEmbeddingsEncoder(args)

        self.context_encoder = SelfAttentiveEncoder(args)

        self.label_encoder = BagOfEmbeddingsEncoder(args)


        cls_input_dim = args.word_dim + 2*args.hidden_dim

        self.repre_proj = nn.Linear(cls_input_dim,args.hidden_dim,bias=False)

        self.classifier = nn.Linear(1,1)

        self.hidden_dim = args.hidden_dim

        self.n_labels = args.n_labels

        self.use_position_embedding = args.use_position_embedding
        self.padding_idx = args.padding_idx

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.parameters(),args.lr)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self,mention,left_context,right_context,candidate_labels,feature=None):

        bsize= mention.size()[0]
        mention_mask = (mention != self.padding_idx).float()

        mention_repre = self.word_embedding(mention)
        mention_repre = self.mention_encoder.forward(mention_repre,mask=mention_mask)

        mention_repre = mention_repre

        left_context_lengths = (left_context != self.padding_idx).sum(dim=1).long().to(device)
        right_context_lengths = (right_context != self.padding_idx).sum(dim=1).long().to(device)

        left_context = self.word_embedding(left_context)
        right_context = self.word_embedding(right_context)

        if self.use_position_embedding:
            left_context = add_positional_features(left_context)
            right_context = add_positional_features(right_context)

        context_repre = self.context_encoder(left_context,left_context_lengths,right_context,right_context_lengths)

        mention_repre = self.repre_proj(torch.cat([mention_repre,context_repre],dim=-1)) # bsize * hidden_dim

        mention_repre = self.dropout(mention_repre)

        label_mask = (candidate_labels != self.padding_idx).float()
        n_classes = candidate_labels.size()[1]
        label_len = candidate_labels.size()[2]
        label_repre = self.label_word_embedding(candidate_labels) # bsize * N_CLASSES * hidden_dim
        label_repre = self.label_encoder.forward(label_repre.view(bsize*n_classes,label_len,-1),label_mask.view(bsize*n_classes,label_len)).view(bsize,n_classes,-1)

        l = torch.bmm(label_repre,mention_repre.view(bsize,self.hidden_dim,1)) # bsize * N_CLASSES * 1

        return self.classifier(l).squeeze()


    def train_epoch(self,train_iter):

        self.train()
        total_batch = len(train_iter)
        loss = 0.0
        cur_batch = 1
        for batch in train_iter:
            mention = torch.tensor(batch['mention']).to(device)
            left_context = torch.tensor(batch['left_context']).to(device)
            right_context = torch.tensor(batch['right_context']).to(device)
            candidate_labels = torch.tensor(batch['full_labels']).to(device)
            labels = torch.tensor(batch['labels']).to(device)

            output = self.forward(mention,left_context,right_context,candidate_labels)# bsize * N_CLASSES
            batch_loss = self.loss_fn(output,labels.float())
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            cur_batch += 1

            loss += batch_loss
            print('\r Batch {}/{}, Training Loss:{}'.format(cur_batch,total_batch,loss/cur_batch),end='')

    def evaluate(self,dev_iter):

        self.eval()
        target_labels = []
        pred_labels = []
        for batch in dev_iter:
            mention = torch.tensor(batch['mention']).to(device)
            left_context = torch.tensor(batch['left_context']).to(device)
            right_context = torch.tensor(batch['right_context']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            candidate_labels = torch.tensor(batch['full_labels']).to(device)

            output = self.forward(mention,left_context,right_context,candidate_labels)

            # ensure at least one label
            one_label = output.argmax(dim=1)

            pred = (torch.sigmoid(output)> 0.5)
            target_labels.extend(one_hot_to_labels(labels))
            pred_labels.extend(one_hot_to_labels(pred,one_label))

        true_and_prediction = list(zip(target_labels,pred_labels))
        return strict(true_and_prediction)[2],loose_macro(true_and_prediction)[2],loose_micro(true_and_prediction)[2]



class Vocab(object):
    def __init__(self):
        pass

    def renew_vocab(self,data,name):
        for d in data:
            if d in getattr(self,name):
                continue
            else:
                getattr(self,name)[d] = len(getattr(self,name))

def build_vocab(filenames):

    vocab = Vocab()
    vocab.stoi = {'<pad>':0,'<unk>':1}
    vocab.ltoi = {}
    vocab.lwtoi = {'<pad>':0,'<unk>':1}
    vocab.ftoi = {}
    for filepath in filenames:
        with open(filepath,'r') as f:
            for line in f:
                mention,labels,left_context,right_context= line.rstrip().split('\t')
                vocab.renew_vocab(mention.split(),'stoi')
                vocab.renew_vocab(left_context.split(),'stoi')
                vocab.renew_vocab(right_context.split(),'stoi')
                for label in labels.split():
                    vocab.renew_vocab(label.replace('/',' ').split(),'lwtoi')
                vocab.renew_vocab(labels.split(),'ltoi')

    return vocab

def main(args):

    # filepaths = ['train.tsv','test.tsv','test.tsv']
    # for i in range(len(filepaths)):
    #     filepaths[i] = os.path.join(args.data_dir,filepaths[i])
    #
    # vocab = build_vocab(filepaths)
    # torch.save(vocab,'./data/FIGER/vocab.pkl')
    # vocab = torch.load('./data/FIGER/vocab.pkl',pickle_module=dill)
    # print('Vocab Loaded!')

    # train_dataset = EntityTypingDataset(filepaths[0],vocab,args.batch_size)
    # train_dataset = torch.load('train.dataset',pickle_module=dill)
    # dev_dataset = EntityTypingDataset(filepaths[1],vocab,args.batch_size)


    # for file in os.listdir(os.path.join(args.data_dir)):
    #     if file.startswith('train.split.'):
    #         train_parts = EntityTypingDataset(os.path.join(args.data_dir,file),vocab,args.batch_size)
    #         print('\n')
    #         torch.save(train_parts,os.path.join('/home/user_data55/lijh/data/FIGER',file+'.pkl'),pickle_module=dill)
    #         print(file)
    #
    # print('Write train file splits.')
    #test_dataset = EntityTypingDataset(filepaths[2],vocab,args.batch_size)

    # torch.save(train_dataset,'/home/user_data55/lijh/FIGER/train.pickle',pickle_module=dill)
    # torch.save(test_dataset,'/home/user_data55/lijh/FIGER/test.pickle',pickle_module=dill)

    # train_parts = []
    # for file in os.listdir(args.data_dir):
    #     if 'train' in file and 'pkl' in file:
    #         train_parts.append(torch.load(os.path.join(args.data_dir,file),pickle_module=dill))
    # train_dataset = torch.utils.data.ConcatDataset(train_parts)
    # torch.save(train_dataset,'./data/FIGER/train.pkl',pickle_module=dill)
    # torch.save(test_dataset,'./data/FIGER/test.pkl',pickle_module=dill)

    import time
    start_time = time.time()
    vocab = torch.load('./data/FIGER/vocab.pkl',pickle_module=dill)
    train_dataset = torch.load('./data/FIGER/train.pkl',pickle_module=dill)
    dev_dataset = torch.load('./data/FIGER/test.pkl',pickle_module=dill)
    test_dataset = torch.load('./data/FIGER/test.pkl',pickle_module=dill)
    end_time = time.time()
    print( 'Loaded dataset in {:.2f}s '.format(end_time-start_time))


    train_iter = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,collate_fn=collate_fn)
    dev_iter = DataLoader(dataset=dev_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,collate_fn=collate_fn)
    test_iter = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,collate_fn=collate_fn)

    args.n_words = len(vocab.stoi)
    args.n_labels = len(vocab.ltoi)
    args.n_labelwords = len(vocab.lwtoi)

    args.padding_idx = 0

    if args.word_pretrained_path is not None:
        if args.glove_pth is not None:
            args.word_pretrained = load_pretrained(args.glove_pth,vocab.stoi,dim=args.word_dim,device=device,pad_idx=args.padding_idx)
            torch.save(args.word_pretrained,args.word_pretrained_path)
        else:
            args.word_pretrained = torch.load(args.word_pretrained_path)

        # args.word_pretrained = torch.load(args.word_pretrained_path,)
    if args.label_word_pretrained_path is not None:
        if args.glove_pth is not None:
            args.label_word_pretrained = load_pretrained(args.glove_pth,vocab.lwtoi,dim=args.label_word_dim,device=device,pad_idx=args.padding_idx)
            torch.save(args.label_word_pretrained,args.label_word_pretrained_path)
        else:
            args.label_word_pretrained = torch.load(args.label_word_pretrained_path)

    model = Model(args).to(device)

    best_acc = -1.
    for epoch in range(args.epoch):
        model.train_epoch(train_iter)
        # train_acc = model.evaluate(train_iter)
        dev_acc = model.evaluate(dev_iter)

        print(' \nEpoch {}, Dev Acc : ({:.2f},{:.2f},{:.2f})'.format(epoch,dev_acc[0],dev_acc[1],dev_acc[2]))
        if dev_acc[0] > best_acc:
            best_acc = dev_acc[0]
            torch.save(model.state_dict(),'model.pth')

    model = Model(args).to(device)
    model.load_state_dict(torch.load('model.pth'))

    dev_acc = model.evaluate(dev_iter)
    test_acc = model.evaluate(test_iter)
    print('Dev Acc: ({:.2f},{:.2f},{:.2f}), Test Acc :({:.2f},{:.2f},{:.2f})'.format(dev_acc[0],dev_acc[1],dev_acc[2],test_acc[0],test_acc[1],test_acc[2]))

class TestConfig:

    def __init__(args):
        args.word_dim = 30
        args.hidden_dim = 30
        args.attention_dim = 10
        args.label_dim = 10
        args.label_word_dim = 30
        args.batch_size = 10
        args.epoch = 20
        args.use_position_embedding = False
        args.data_dir = '../../data/FIGER'
        # args.pretrained_path = '/home/user_data/lijh/data/english_embeddings/glove.840B.300d.txt'
        args.word_pretrained_path = None
        args.label_word_pretrained_path = None

        args.lr = 1e-3
        args.pretrained = None

class DefaultConfig:

    def __init__(args):
        args.word_dim = 300
        args.hidden_dim = 300
        args.attention_dim = 100
        args.label_dim = 300
        args.label_word_dim = 300
        args.batch_size = 1024
        args.epoch = 5
        args.use_position_embedding = False
        args.data_dir = '/home/user_data55/lijh/data/FIGER'
        # args.glove_pth = '/home/user_data/lijh/data/english_embeddings/glove.840B.300d.txt'
        args.glove_pth = None
        args.word_pretrained_path = './data/FIGER/word_pretrained.pth'
        # args.word_pretrained_path = None
        args.label_word_pretrained_path = './data/FIGER/label_pretrained.pth'
        # args.label_word_pretrained_path = None
        args.lr = 1e-3

if __name__ == '__main__':
    import sys
    if sys.argv[1] == '--test':
        args = TestConfig()
    elif sys.argv[1] == '--train':
        args= DefaultConfig()
    main(args)
