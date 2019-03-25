import torch
import dgl
import dill
import os
import numpy as np
from torch.utils.data import DataLoader
from dataloader.simpleQA_dataloader import SimpleQADataset
from learner.SimpleQA import SimpleQA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_folds(labelset,K=10):
    length = len(labelset)
    len_of_each_folds = length // K
    label_list = list(labelset)
    folds = []
    for i in range(0,length,len_of_each_folds):
        folds.append(label_list[i:min(i + len_of_each_folds,length)])
    return folds


def main(args):

    import time
    start_time = time.time()
    vocab = SimpleQADataset.load_vocab(args)
    adj_matrix = torch.load(args.relation_adj_matrix_pth)
    print('Relation Adj matrix loaded!')
    num_nodes = len(adj_matrix)
    print('Total number of relations :{}'.format(num_nodes))
    src = []
    dst = []
    triplets = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                src.append(i)
                dst.append(j)
                triplets.append((i,j))
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src,dst)

    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
        norm = 1.0 / in_deg
        norm[np.isinf(norm)] = 0
        return norm
    norm = comp_deg_norm(g)
    node_id = torch.arange(0,num_nodes,dtype=torch.long).view(-1,1).to(device)
    norm = torch.from_numpy(norm).view(-1,1).to(device)
    g.ndata.update({'id':node_id,'norm':norm})
    print('Building Relation Graph ... Done.')
    args.relation_graph = g
    args.relation_pretrained = torch.load(args.relation_pretrained_pth)
    end_time = time.time()
    print('Loaded dataset in {:.2f}s'.format(end_time - start_time))

    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)
    args.all_relation_words = vocab.get_all_relation_words()

    if args.word_pretrained_pth is not None:
        args.word_pretrained = torch.load(args.word_pretrained_pth)
        print(' Pretrained word embedding loaded!')
    else:
        args.word_pretrained = None
        print(' Using random initialized word embedding.')

    if args.relation_pretrained_pth is not None:
        args.relation_pretrained = torch.load(args.relation_pretrained_pth)
        print(' Pretrained label word embedding loaded!')
    else:
        args.relation_pretrained = None
        print(' Using random initialized label word embedding.')

    if args.mode == 'supervised':
        train_dataset,dev_dataset,test_dataset = SimpleQADataset.load_dataset(args)
        train(args,train_dataset,dev_dataset,test_dataset,vocab,SimpleQADataset.collate_fn)
    elif args.mode == 'zero-shot':
        base_data_dir = args.data_dir
        base_save_dir = args.save_dir
        for i in range(10):
            print(' Training Fold {}'.format(i))
            train_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'train.tsv')
            dev_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'dev.tsv')
            test_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'test.tsv')
            train_dataset,dev_dataset,test_dataset = SimpleQADataset.load_dataset(train_fname,dev_fname,test_fname,args.vocab_pth,args)
            args.save_dir = os.path.join(base_save_dir,'fold-{}'.format(str(i)))
            train(args,train_dataset,dev_dataset,test_dataset,vocab,SimpleQADataset.collate_fn)


def train(args,train_dataset,dev_dataset,test_dataset,vocab,collate_fn):

    train_iter = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=False,num_workers=12,collate_fn=collate_fn)
    dev_iter = DataLoader(dataset=dev_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=collate_fn)
    test_iter = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=collate_fn)

    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)

    args.padding_idx = 0

    print('Building Model...',end='')
    model = SimpleQA(args).to(device)
    print('Done')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    best_acc = -1.
    patience = args.patience
    test_acc = -1.
    logfile = open(os.path.join(args.save_dir,'log.txt'),'w')
    for epoch in range(args.epoch):
        model.train_epoch(train_iter)
        dev_acc = model.evaluate(dev_iter)
        patience -= 1

        print(' \nEpoch {}, Patience : {}, Dev Acc : {:.2f}'.format(epoch,patience,dev_acc*100))
        print(' \nEpoch {}, Patience : {}, Dev Acc : {:.2f}'.format(epoch,patience,dev_acc*100),file=logfile)
        if patience > 0 and dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(),os.path.join(args.save_dir,'model.pth'))
            patience = args.patience

        if patience == 0:
            model = SimpleQA(args).to(device)
            model.load_state_dict(torch.load(os.path.join(args.save_dir,'model.pth')))
            dev_acc = model.evaluate(dev_iter)
            test_acc = model.evaluate(test_iter)
            print('Dev Acc: {:.2f}, Test Acc :{:.2f}'.format(dev_acc*100,test_acc*100))
            print('Dev Acc: {:.2f}, Test Acc :{:.2f}'.format(dev_acc*100,test_acc*100),file=logfile)
            logfile.close()
            return test_acc


class DefaultConfig:
    def __init__(self):
        self.word_dim = 300
        self.hidden_dim = 150
        self.relation_dim = 300
        self.batch_size = 64
        self.epoch = 100
        self.data_dir = None
        self.lr = 1e-3
        self.margin = 0.1
        self.ns = 256
        self.patience = 5

        self.data_dir = './data/SimpleQuestions/10-fold-dataset-tsv'
        self.vocab_pth = os.path.join(self.data_dir,'vocab.pth')
        self.relation_file = os.path.join(self.data_dir,'relation.id')
        self.word_pretrained_pth = os.path.join(self.data_dir,'word_pretrained.pth')

        self.freeze = True
        self.num_bases = 100
        self.num_hidden_layers = 4
        self.rgcn_dropout = 0.0

        self.padding_idx = 0
        self.pad_token = '<pad>'
        self.unk_idx = 1
        self.unk_token = '<unk>'

        self.save_dir = 'results/simpleQA/gcn'

        self.glove_pth = '/home/user_data/lijh/data/english_embeddings/glove.6B.300d.txt'
        self.relation_vec_pth = './data/SimpleQuestions/relation2vec.txt'

        self.relation_pretrained_pth = './data/SimpleQuestions/relation_pretrained.pth'
        self.relation_adj_matrix_pth = './data/SimpleQuestions/relation_adj_matrix.pth'

        self.threshold = 10

        import pprint
        pprint.pprint(self.__dict__)


class TestConfig:
    def __init__(self):
        self.word_dim = 3
        self.hidden_dim = 3
        self.relation_dim = 6
        self.batch_size = 10
        self.epoch = 5
        self.data_dir = 'data/SimpleQA'
        self.lr = 1e-3
        self.ns = 4
        self.padding_idx = 0
        self.pad_token = '<pad>'
        self.unk_idx = 1
        self.unk_token = '<unk>'
        self.relation_file = 'data/SimpleQA/FB2M.rel_voc.pickle'
        self.vocab_pth = 'data/SimpleQA/vocab.pth'

        self.train_dataset_pth = './data/SimpleQA/train.pkl'
        self.dev_dataset_pth = './data/SimpleQA/dev.pkl'
        self.test_dataset_pth = './data/SimpleQA/test.pkl'

        self.word_pretrained_path = '/home/user_data/lijh/data/glove.6B.300d.txt'
        self.relation_pretrained_path = None


if __name__ == '__main__':
    import sys
    if sys.argv[1] == '--supervised':
        args = DefaultConfig()
        args.mode = 'supervised'
        args.data_dir = './data/SimpleQuestions'
        args.vocab_pth = os.path.join(args.data_dir,'vocab.pth')
        args.relation_file = os.path.join(args.data_dir,'relation.id')
        args.word_pretrained_pth = os.path.join(args.data_dir,'word_pretrained.pth')
        main(args)
    elif sys.argv[1] == '--test':
        args = TestConfig()
        main(args)
    elif sys.argv[1] == '--zero-shot':
        args = DefaultConfig()
        args.mode = 'zero-shot'
        args.data_dir = './data/SimpleQuestions/10-fold-dataset-tsv'
        args.vocab_pth = os.path.join(args.data_dir,'vocab.pth')
        args.relation_file = os.path.join(args.data_dir,'relation.id')
        args.word_pretrained_pth = os.path.join(args.data_dir,'word_pretrained.pth')
        main(args)
    elif sys.argv[1] == '--generate':
        args = DefaultConfig()
        args.data_dir = os.path.join(args.data_dir,'base')
        SimpleQADataset.generate_vocab(args)
        SimpleQADataset.generate_embedding(args,device)
        SimpleQADataset.generate_graph(args,device)
