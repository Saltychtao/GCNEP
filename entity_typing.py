import sys
sys.path.append('..')


import os
from collections import defaultdict
import torch
import torch.nn as nn
import dill
import linecache


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

    def __init__(self,filename,vocab,batch_size,share_vocab):

        self.share_vocab = share_vocab
        self.vocab = vocab
        self.dataset = self.read_file(filename,vocab)
        self.filename = filename
        self.batch_size = batch_size

    def read_file(self,filename,vocab):
        dataset = []
        self.full_label_set = list(
            map(lambda x: x.replace('/', ' ').replace('_',' ').split(), list(sorted(vocab.ltoi, key=vocab.ltoi.get))))
        self.label_dict = defaultdict(lambda : [])
        self.label_set = set()
        self.length = 0
        cnt = 0
        with open(filename,'r') as f:
            for line in f:
                mention,labels,left_context,right_context,_= line.rstrip().split('\t')
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

                for label in labels.split():
                    self.label_dict[label].append(cnt)
                    self.label_set.add(label)
                dataset.append(instance)
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

    def get_subset(self,labels,mode):
        idxs = []
        if mode == 'seen':
            for label in labels:
                idxs.extend(self.label_dict[label])

        elif mode == 'unseen':
            for key,val in self.label_dict.items():
                if key not in labels:
                    idxs.extend(val)
        return torch.utils.data.Subset(self,list(set(idxs)))

    def __len__(self):
        return self.length

    def __getitem__(self,item):
        line = linecache.getline(self.filename,item+1)
        instance = self.process_line(line)
        instance['labels_idx'] = scatter([self.vocab.ltoi[label] for label in instance['labels']],
                                     len(self.vocab.ltoi))
        if self.share_vocab:
            instance['full_labels'] = [[self.vocab.stoi[word] for word in label] for label in
                                       self.full_label_set]
        else:
            instance['full_labels'] = [[self.vocab.lwtoi[word] for word in label] for label in
                                       self.full_label_set]
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
    labels = [x['labels_idx'] for x in list_of_examples]
    full_labels = pad(list_of_examples[0]['full_labels'],0)

    batch['mention'] = np.array(mention)
    batch['left_context'] = np.array(left_context)
    batch['right_context'] = np.array(right_context)
    batch['full_labels'] = np.array([full_labels for i in range(len(list_of_examples))])
    batch['labels'] = np.array(labels)

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
            self.word_embedding = nn.Embedding.from_pretrained(args.word_pretrained,freeze=args.freeze)

        self.label_embedding = nn.Embedding(args.n_labels,args.label_dim)

        if not args.share_vocab:
            if args.label_word_pretrained is None:
                self.label_word_embedding = nn.Embedding(args.n_labelwords,args.label_word_dim,args.padding_idx)
            else:
                self.label_word_embedding = nn.Embedding.from_pretrained(args.label_word_pretrained,freeze=args.freeze)

        self.mention_encoder = BagOfEmbeddingsEncoder(args)

        self.context_encoder = SelfAttentiveEncoder(args)

        self.label_encoder = BagOfEmbeddingsEncoder(args)

        cls_input_dim = args.word_dim + 2*args.hidden_dim

        self.repre_proj = nn.Sequential(
            nn.Linear(cls_input_dim,args.label_word_dim,bias=False),
            nn.ReLU()
        )
        self.hidden_dim = args.hidden_dim
        self.label_word_dim = args.label_word_dim

        self.n_labels = args.n_labels

        self.share_voacb = args.share_vocab

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

        mention_repre = self.dropout(mention_repre)

        left_context_lengths = (left_context != self.padding_idx).sum(dim=1).long().to(device)
        right_context_lengths = (right_context != self.padding_idx).sum(dim=1).long().to(device)

        left_context = self.word_embedding(left_context)
        right_context = self.word_embedding(right_context)

        if self.use_position_embedding:
            left_context = add_positional_features(left_context)
            right_context = add_positional_features(right_context)

        context_repre = self.context_encoder(left_context,left_context_lengths,right_context,right_context_lengths)

        mention_repre = self.repre_proj(torch.cat([mention_repre,context_repre],dim=-1)) # bsize * hidden_dim

        label_mask = (candidate_labels != self.padding_idx).float()
        n_classes = candidate_labels.size()[1]
        label_len = candidate_labels.size()[2]

        if self.share_voacb:
            label_repre = self.word_embedding(candidate_labels)
        else:
            label_repre = self.label_word_embedding(candidate_labels) # bsize * N_CLASSES * hidden_dim
        label_repre = self.label_encoder.forward(label_repre.view(bsize*n_classes,label_len,-1),label_mask.view(bsize*n_classes,label_len)).view(bsize,n_classes,-1)

        # single_label_repre = self.label_embedding(torch.tensor([[i for i in range(n_classes)] * bsize]).to(device)).view(bsize,n_classes,-1)

        score = torch.bmm(label_repre,mention_repre.view(bsize,self.label_word_dim,1)) # bsize * N_CLASSES * 1

        return score.squeeze()


    def train_epoch(self,train_iter):

        self.train()
        total_batch = len(train_iter)
        loss = 0.0
        cur_batch = 1
        for batch in train_iter:
            mention = torch.from_numpy(batch['mention']).to(device)
            left_context = torch.from_numpy(batch['left_context']).to(device)
            right_context = torch.from_numpy(batch['right_context']).to(device)
            candidate_labels = torch.from_numpy(batch['full_labels']).to(device)
            labels = torch.from_numpy(batch['labels']).to(device)

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

def generate_dataset(args):

    filepaths = ['train.tsv', 'dev.tsv', 'test.tsv']
    for i in range(len(filepaths)):
        filepaths[i] = os.path.join(args.data_dir, filepaths[i])

    vocab = build_vocab(filepaths, args)
    torch.save(vocab, args.vocab_pth)

    print('Saved vocab')

    train_dataset = EntityTypingDataset(filepaths[0],vocab,args.batch_size,args.share_vocab)
    dev_dataset = EntityTypingDataset(filepaths[1],vocab,args.batch_size,args.share_vocab)
    test_dataset = EntityTypingDataset(filepaths[2], vocab, args.batch_size, args.share_vocab)
    print('\n Saving datasets.')
    torch.save(train_dataset, args.train_dataset_pth, pickle_module=dill)
    torch.save(dev_dataset,args.dev_dataset_pth,pickle_module=dill)
    torch.save(test_dataset, args.test_dataset_pth, pickle_module=dill)


def generate_embedding(args):
    vocab = torch.load(args.vocab_pth)
    if args.word_pretrained_path is not None:
        if args.glove_pth is not None:
            args.word_pretrained = load_pretrained(args.glove_pth, vocab.stoi, dim=args.word_dim, device=device,
                                                   pad_idx=args.padding_idx)
            torch.save(args.word_pretrained, args.word_pretrained_path)
        else:
            args.word_pretrained = torch.load(args.word_pretrained_path)

    if args.label_word_pretrained_path is not None and not args.share_vocab:
        if args.glove_pth is not None:
            args.label_word_pretrained = load_pretrained(args.glove_pth, vocab.lwtoi, dim=args.label_word_dim,
                                                         device=device, pad_idx=args.padding_idx)
            torch.save(args.label_word_pretrained, args.label_word_pretrained_path)
        else:
            args.label_word_pretrained = torch.load(args.label_word_pretrained_path)


def generate_folds(labelset,K=10):
    length = len(labelset)
    len_of_each_folds = length // K
    label_list = list(labelset)
    folds = []
    for i in range(0,length,len_of_each_folds):
        folds.append(label_list[i:min(i+len_of_each_folds,length)])
    return folds


def main(args):
    import time
    start_time = time.time()
    vocab = torch.load(args.vocab_pth,pickle_module=dill)
    # train_dataset = torch.load(args.train_dataset_pth,pickle_module=dill)
    # dev_dataset = torch.load(args.dev_dataset_pth,pickle_module=dill)
    # test_dataset = torch.load(args.test_dataset_pth,pickle_module=dill)
    filepaths = ['train.tsv', 'dev.tsv', 'test.tsv']
    for i in range(len(filepaths)):
        filepaths[i] = os.path.join(args.data_dir, filepaths[i])
    train_dataset = EntityTypingDataset(filepaths[0],vocab,args.batch_size,args.share_vocab)
    dev_dataset = EntityTypingDataset(filepaths[1],vocab,args.batch_size,args.share_vocab)
    test_dataset = EntityTypingDataset(filepaths[2], vocab, args.batch_size, args.share_vocab)
    end_time = time.time()
    print('Loaded dataset in {:.2f}s '.format(end_time-start_time))

    if args.word_pretrained_path is not None:
        args.word_pretrained = torch.load(args.word_pretrained_path)
        print(' Pretrained word embedding loaded!')
    else:
        args.word_pretrained = None
        print(' Using random initialized word embedding.')

    if args.label_word_pretrained_path is not None and not args.share_vocab:
        args.label_word_pretrained = torch.load(args.label_word_pretrained_path)
        print(' Pretrained label word embedding loaded!')
    else:
        args.label_word_pretrained = None
        print(' Using random initialized label word embedding.')

    if args.mode == 'zero-shot':
        train_labels = train_dataset.get_label_set()
        test_labels = test_dataset.get_label_set()
        folds = generate_folds(test_labels)

        for i,fold in enumerate(folds):
            print('Training Folds: {}'.format(i))
            train_subset = train_dataset.get_subset(fold,mode='unseen')
            dev_subset = dev_dataset.get_subset(fold,mode='unseen')
            test_subset = test_dataset.get_subset(fold,mode='seen')

            with open('fold/fold{}.txt'.format(str(i)),'w') as f:
                f.write('Training Subset labels: {}'.format(' '.join(list(train_labels - set(fold)))) + '\n')
                f.write('Test Subset labels: {}'.format(' '.join(fold)))

            # print('Total Training Instances :{}'.format(len(train_subset)))
            # print('Total Training labels :{}'.format(len(train_subset.get_label_set())))
            # print('Total Test Instances: {}'.format(len(test_subset)))

            test_acc = train(args,train_subset,dev_subset,test_subset,vocab)
            with open('fold/fold{}.txt'.format(str(i)),'a') as f:
                f.write('Test Acc :({:.2f},{:.2f},{:.2f}'.format(test_acc[0],test_acc[1],test_acc[2]))

    elif args.mode == 'supervised':
        train(args,train_dataset,dev_dataset,test_dataset,vocab)


def train(args,train_dataset,dev_dataset,test_dataset,vocab):

    train_iter = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=False,num_workers=8,collate_fn=collate_fn)
    dev_iter = DataLoader(dataset=dev_dataset,batch_size=args.batch_size,shuffle=False,num_workers=8,collate_fn=collate_fn)
    test_iter = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=8,collate_fn=collate_fn)

    args.n_words = len(vocab.stoi)
    args.n_labels = len(vocab.ltoi)
    args.n_labelwords = len(vocab.lwtoi)

    args.padding_idx = 0

    model = Model(args).to(device)

    best_acc = -1.
    for epoch in range(args.epoch):
        model.train_epoch(train_iter)
        dev_acc = model.evaluate(dev_iter)

        print(' \nEpoch {}, Dev Acc : ({:.2f},{:.2f},{:.2f})'.format(epoch,dev_acc[0],dev_acc[1],dev_acc[2]))
        if dev_acc[0] > best_acc:
            best_acc = dev_acc[0]
            torch.save(model.state_dict(),args.save_pth)

    model = Model(args).to(device)
    model.load_state_dict(torch.load(args.save_pth))

    dev_acc = model.evaluate(dev_iter)
    test_acc = model.evaluate(test_iter)
    print('Dev Acc: ({:.2f},{:.2f},{:.2f}), Test Acc :({:.2f},{:.2f},{:.2f})'.format(dev_acc[0],dev_acc[1],dev_acc[2],test_acc[0],test_acc[1],test_acc[2]))
    return test_acc

class TestConfig:

    def __init__(args):
        args.word_dim = 30
        args.hidden_dim = 30
        args.attention_dim = 10
        args.label_dim = 10
        args.label_word_dim = 30
        args.batch_size = 10
        args.share_vocab =True
        args.epoch = 20
        args.use_position_embedding = False
        args.data_dir = '../../data/FIGER-gold'
        # args.pretrained_path = '/home/user_data/lijh/data/english_embeddings/glove.840B.300d.txt'
        args.word_pretrained_path = None
        args.label_word_pretrained_path = None
        args.word_pretrained = None
        args.label_word_pretrained = None

        args.padding_idx = 0
        args.lr = 1e-3

        args.vocab_pth = './data/FIGER/share_vocab/vocab.pkl'
        args.train_dataset_pth = './data/FIGER/share_vocab/train.pkl'
        args.dev_dataset_pth = './data/FIGER/share_vocab/dev.pkl'
        args.test_dataset_pth = './data/FIGER/share_vocab/test.pkl'

class DefaultConfig:

    def __init__(args):
        args.word_dim = 300
        args.hidden_dim = 100
        args.attention_dim = 50
        args.label_dim = 300
        args.label_word_dim = 300
        args.batch_size = 1024
        args.epoch = 10
        args.use_position_embedding = False
        args.share_vocab = True
        args.data_dir = '/home/user_data55/lijh/data/FIGER'

        args.word_pretrained_path = './data/FIGER/share_vocab/word_pretrained.pth'
        args.label_word_pretrained_path = './data/FIGER/share_vocab/label_pretrained.pth'
        args.lr = 1e-3
        args.padding_idx = 0

        args.vocab_pth = './data/FIGER/share_vocab/vocab.pkl'
        args.train_dataset_pth = './data/FIGER/share_vocab/train.pkl'
        args.dev_dataset_pth = './data/FIGER/share_vocab/dev.pkl'
        args.test_dataset_pth = './data/FIGER/share_vocab/test.pkl'

        args.save_pth = 'zero_shot.pth'

        args.freeze = True

if __name__ == '__main__':
    import sys
    if sys.argv[1] == '--test':
        args = TestConfig()
        args.glove_pth = None
        args.mode = 'supervised'
        main(args)
    elif sys.argv[1] == '--supervised':
        args = DefaultConfig()
        args.mode = 'supervised'
        main(args)
    elif sys.argv[1] == '--zero-shot':
        args = DefaultConfig()
        args.mode = 'zero-shot'
        main(args)
    elif sys.argv[1] == '--generate':
        args = DefaultConfig()
        args.glove_pth = '/home/user_data/lijh/data/english_embeddings/glove.840B.300d.txt'
        generate_dataset(args)
        generate_embedding(args)
