import sys
sys.path.append('..')

from torchtext import data
import os
import torch
import torch.nn as nn

import numpy as np
from lib.module import LSTMEncoder
from lib.utils.util import load_pretrained
from allennlp.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp.nn.util import add_positional_features,weighted_sum

from evaluate  import strict,loose_macro,loose_micro
from torch.utils.data import Dataset, DataLoader



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


class EntityTypingDataset(Dataset):

    def __init__(self,filename,vocab,batch_size):

        raw_dataset = self.read_file(filename)
        self.dataset = self.numeralize(raw_dataset,vocab)
        self.batch_size = batch_size
        self.vocab = vocab

    def read_file(self,filename):
        dataset = []
        with open(filename,'r') as f:
            for line in f:
                mention,labels,left_context,right_context,features= line.rstrip().split('\t')
                example = dict()
                example['mention'] = mention.split()
                example['labels'] = labels.split()
                example['left_context'] = left_context.split()
                example['right_context'] = right_context.split()
                features['features'] = features.split()

                dataset.append(example)
        return dataset

    def numeralize(self,raw_dataset,vocab):
        dataset = []
        for example in raw_dataset:
            mention = [vocab.stoi[word] for word in example['mention']]
            labels = [vocab.ltoi[label] for label in example['labels']]
            left_context = [vocab.stoi[word] for word in example['left_context']]
            right_context = [vocab.stoi[word] for word in example['right_context']]
            features = [vocab.ftoi[feature] for feature in example['features']]

            instance = dict()
            instance['mention'] = mention
            instance['labels'] = labels
            instance['left_context'] = left_context
            instance['right_context'] = right_context
            instance['features'] = features
            dataset.append(instance)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,item):
        return self.dataset[item]


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

    batch['mention'] = mention
    batch['left_context'] = left_context
    batch['right_context'] = right_context
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

        if args.pretrained is None:
            self.word_embedding = nn.Embedding(args.n_words,args.word_dim,args.padding_idx)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(args.pretrained,freeze=True)

        self.feature_proj = nn.Linear(args.feature_dim,args.hidden_dim)

        self.label_embedding = nn.Embedding(args.n_labels,args.label_dim)

        self.mention_encoder = BagOfEmbeddingsEncoder(args)

        self.context_encoder = SelfAttentiveEncoder(args)

        if not args.use_feature:
            cls_input_dim = args.word_dim + 2*args.hidden_dim
        else:
            cls_input_dim = args.word_dim + 2*args.hidden_dim + args.feature_dim

        self.repre_proj = nn.Linear(cls_input_dim,args.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(1,1),
            nn.ReLU(),
            nn.Linear(1,1)
        )

        self.hidden_dim = args.hidden_dim

        self.n_labels = args.n_labels

        self.use_position_embedding = args.use_position_embedding
        self.padding_idx = args.padding_idx

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self,mention,context_l,context_r,feature):

        bsize= mention.size()[0]
        mention_mask = (mention != self.padding_idx).float()

        mention_repre = self.word_embedding(mention)
        mention_repre = self.mention_encoder.forward(mention_repre,mask=mention_mask)

        mention_repre = self.dropout(mention_repre)

        context_l,context_l_lengths = context_l
        context_r,context_r_lengths = context_r

        context_repre_l = self.word_embedding(context_l)
        context_repre_r = self.word_embedding(context_r)

        if self.use_position_embedding:
            context_repre_l = add_positional_features(context_repre_l)
            context_repre_r = add_positional_features(context_repre_r)

        context_repre = self.context_encoder(context_repre_l,context_l_lengths,context_repre_r,context_r_lengths)

        mention_repre = self.repre_proj(torch.cat([mention_repre,context_repre],dim=-1)) # bsize * hidden_dim

        label_repre = self.label_embedding(torch.tensor([[i for i in range(self.n_labels)] for j in range(bsize)]).to(device)) # bsize * N_CLASSES * hidden_dim

        l = torch.bmm(label_repre,mention_repre.view(bsize,self.hidden_dim,1)) # bsize * N_CLASSES * 1

        return self.classifier(l).squeeze()


    def train_epoch(self,train_iter,):

        self.train()
        total_batch = len(train_iter)
        loss = 0.0
        cur_batch = 1
        for batch in train_iter:
            mention,mention_lengths = batch.mention
            labels,labels_lengths = batch.labels #  bsize * N_CLASSES
            features,features_lengths = batch.features

            output = self.forward(mention,batch.left_context,batch.right_context,features)# bsize * N_CLASSES
            batch_loss = self.loss_fn(output,labels.float())
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            cur_batch += 1

            loss += batch_loss
            print('\r Batch {}/{}, Training Loss:{:.2f}'.format(cur_batch,total_batch,loss/cur_batch),end='')

    def evaluate(self,dev_iter):

        self.eval()
        target_labels = []
        pred_labels = []
        for batch in dev_iter:
            mention,mention_lengths = batch.mention
            labels,labels_lengths = batch.labels #  bsize * N_CLASSES
            features,features_lengths = batch.features

            output = self.forward(mention,batch.left_context,batch.right_context,features)

            # ensure at least one label
            one_label = output.argmax(dim=1)

            pred = (torch.sigmoid(output)> 0.5)
            target_labels.extend(one_hot_to_labels(labels))
            pred_labels.extend(one_hot_to_labels(pred,one_label))

        true_and_prediction = list(zip(target_labels,pred_labels))
        return strict(true_and_prediction)[2],loose_macro(true_and_prediction)[2],loose_micro(true_and_prediction)[2]



def load_dataset(text_field,label_field,feature_field,batch_size,data_dir):
    train = data.TabularDataset(path=os.path.join(data_dir,'train.tsv'),
                                format='tsv',
                                fields=[('mention',text_field),('labels',label_field),('left_context',text_field),('right_context',text_field),('features',feature_field)],
    )
    dev = data.TabularDataset(path=os.path.join(data_dir,'dev.tsv'),
                              format='tsv',
                              fields=[('mention',text_field),('labels',label_field),('left_context',text_field),('right_context',text_field),('features',feature_field)])

    test = data.TabularDataset(path=os.path.join(data_dir,'test.tsv'),
                               format='tsv',
                               fields=[('mention',text_field),('labels',label_field),('left_context',text_field),('right_context',text_field),('features',feature_field)])

    text_field.build_vocab(train,dev,test)
    label_field.build_vocab(train,dev,test)
    feature_field.build_vocab(train,dev,test)

    train_iter = data.BucketIterator(train,
                                     train=train,
                                     batch_size=batch_size,
                                     sort_key=lambda x:len(x.left_context),
                                     device=device,
                                     sort_within_batch=True,
                                     repeat=False)

    dev_iter = data.BucketIterator(dev,
                                   batch_size=batch_size,
                                   train=False,
                                   device=device,
                                   sort=False,
                                   shuffle=False,
                                   repeat=False)

    test_iter = data.BucketIterator(test,
                                    batch_size=batch_size,
                                    train=False,
                                    device=device,
                                    sort=False,
                                    shuffle=False,
                                    repeat=False)
    return train_iter,dev_iter,test_iter



def main(args):



    text_field = data.Field(batch_first=True,include_lengths=True)
    label_field = data.Field(batch_first=True,include_lengths=True,postprocessing=postprocessing)
    feature_field = data.Field(batch_first=True,include_lengths=True)
    train_iter, dev_iter, test_iter = load_dataset(text_field,label_field,feature_field,args.batch_size,args.data_dir)

    args.n_words = len(text_field.vocab.stoi)
    args.n_labels = len(label_field.vocab.stoi) - 2
    args.padding_idx = text_field.vocab.stoi[text_field.pad_token]

    args.pretrained = None

    if args.pretrained_path is not None:
        #args.pretrained = load_pretrained(args.pretrained_path,text_field.vocab,dim=args.word_dim,device=device,pad_idx=args.padding_idx)
        #torch.save(args.pretrained,'glove.pth')
        args.pretrained = torch.load(args.pretrained_path,)

    model = Model(args).to(device)

    best_acc = 0.
    for epoch in range(args.epoch):
        model.train_epoch(train_iter)
        dev_acc = model.evaluate(dev_iter)
        print(' \nEpoch {}, Dev Acc : ({:.2f},{:.2f},{:.2f})'.format(epoch,dev_acc[0],dev_acc[1],dev_acc[2]))
        if dev_acc[0] > best_acc:
            best_acc = dev_acc[0]
            torch.save(model.state_dict(),'model.pth')
    model.load_state_dict(torch.load('model.pth'))

    dev_acc = model.evaluate(dev_iter)
    test_acc = model.evaluate(test_iter)
    print('Dev Acc: ({:.2f},{:.2f},{:.2f}), Test Acc :({:.2f},{:.2f},{:.2f})'.format(dev_acc[0],dev_acc[1],dev_acc[2],test_acc[0],test_acc[1],test_acc[2]))

class TestConfig:

    def __init__(args):
        args.word_dim = 30
        args.hidden_dim = 10
        args.attention_dim = 10
        args.feature_dim = 10
        args.label_dim = 10
        args.batch_size = 10
        args.epoch = 5
        args.use_position_embedding = False
        args.use_feature = False
        args.data_dir = '../../data/OntoNotes-processed'
        # args.pretrained_path = '/home/user_data/lijh/data/english_embeddings/glove.840B.300d.txt'
        args.pretrained_path = None

        args.pretrained = None

class DefaultConfig:

    def __init__(args):
        args.word_dim = 300
        args.hidden_dim = 300
        args.attention_dim = 100
        args.feature_dim = 10
        args.label_dim = 300
        args.batch_size = 1024
        args.epoch = 5
        args.use_position_embedding = False
        args.use_feature = False
        args.data_dir = '../../data/OntoNotes-processed'
        # args.pretrained_path = '/home/user_data/lijh/data/english_embeddings/glove.840B.300d.txt'
        args.pretrained_path = 'glove.pth'

        args.pretrained = None


def postprocessing(arr,vocab):
    matrix = []
    for a in arr:
        vec = [0 for i in range(len(vocab.stoi) - 2)]
        for l in a:
            if l > 1:
                vec[l-2] = 1
        matrix.append(vec)
    return matrix

if __name__ == '__main__':
    import sys
    if sys.argv[1] == '--test':
        args = TestConfig()
    elif sys.argv[1] == '--train':
        args= DefaultConfig()
    main(args)
