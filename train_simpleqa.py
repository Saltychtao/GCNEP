import torch
import os
import numpy as np
from pprint import pprint
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from dataloader.simpleQA_dataloader import SimpleQADataset
from learner.SimpleQA import SimpleQA
from utils.graph_util import build_graph_from_adj_matrix
from utils.visualize import plot
from utils.util import parse_args

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
    end_time = time.time()
    print('Loaded dataset in {:.2f}s'.format(end_time - start_time))

    if args.use_gcn:
        args.relation_graphs = []
        for pth in args.relation_adj_matrix_pth:
            adj_matrix = torch.load(pth)
            print('Relation Adj matrix loaded!')
            print('Building Relation Graph ...',end='')
            if not args.self_loop:
                # remove self loop
                print('Removing Self-Loop')
                for i in range(adj_matrix.shape[0]):
                    adj_matrix[i][i] = 0
            g = build_graph_from_adj_matrix(adj_matrix,device)
            print('Done.')
            args.relation_graphs.append(g)
        args.sub_relation_dim = args.relation_dim // len(args.relation_graphs)

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

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    base_data_dir = args.data_dir
    base_save_dir = args.save_dir

    for i in range(args.fold):
        train_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'train.tsv')
        dev_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'dev.tsv')
        test_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'test.tsv')
        train_dataset,dev_dataset,test_dataset = SimpleQADataset.load_dataset(train_fname,dev_fname,test_fname,args.vocab_pth,args)
        args.save_dir = os.path.join(base_save_dir,'fold-{}'.format(str(i)))
        args.log_dir = os.path.join(args.save_dir,'logwriter')
        args.writer = SummaryWriter(log_dir=args.log_dir)

        train_relations = train_dataset.get_label_set()
        args.seen_idx = list(train_relations)
        args.unseen_idx = list(set([i  for i in range(len(vocab.rtoi))]) - set(args.seen_idx))
        label_idx = np.zeros(len(vocab.rtoi))
        label_idx[args.seen_idx] = 1

        if args.train:
            print(' Training Fold {}'.format(i))
            train(args,train_dataset,dev_dataset,test_dataset,vocab,SimpleQADataset.collate_fn)
        elif args.evaluate:
            with torch.no_grad():
                print(' Test Fold {}'.format(i))
                # All
                test_micro_acc,test_macro_acc = evaluate(args,test_dataset,vocab,SimpleQADataset.collate_fn)
                print('Test Acc :({:.2f},{:.2f})'.format(test_micro_acc*100,test_macro_acc*100))

                # Seen
                train_relations = train_dataset.get_label_set()
                seen_subset = test_dataset.get_subset(train_relations,'seen')
                seen_micro_acc,seen_macro_acc = evaluate(args,seen_subset,vocab,SimpleQADataset.collate_fn)
                print('Seen Acc :({:.2f},{:.2f})'.format(seen_micro_acc*100,seen_macro_acc*100))

                # Unseen
                train_relations = train_dataset.get_label_set()
                seen_subset = test_dataset.get_subset(train_relations,'unseen')
                unseen_micro_acc,unseen_macro_acc = evaluate(args,seen_subset,vocab,SimpleQADataset.collate_fn)
                print('UnSeen Acc :({:.2f},{:.2f})'.format(unseen_micro_acc*100,unseen_macro_acc*100))
        elif args.visualize:
            print(' Visualizing Fold {}'.format(i))
            fname = os.path.join(args.save_dir,'embedding.png')
            visualize(args,label_idx,vocab,fname)


def train(args,train_dataset,dev_dataset,test_dataset,vocab,collate_fn):

    train_iter = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=False,num_workers=12,collate_fn=collate_fn)
    dev_iter = DataLoader(dataset=dev_dataset,batch_size=32,shuffle=True,num_workers=12,collate_fn=collate_fn)
    test_iter = DataLoader(dataset=test_dataset,batch_size=32,shuffle=True,num_workers=12,collate_fn=collate_fn)

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
        with torch.no_grad():
            dev_acc = model.evaluate(dev_iter)
        patience -= 1

        print(' \nEpoch {}, Patience : {}, Dev Acc : ({:.2f},{:.2f})'.format(epoch,patience,dev_acc[0]*100,dev_acc[1]*100))
        print(' \nEpoch {}, Patience : {}, Dev Acc : ({:.2f},{:.2f})'.format(epoch,patience,dev_acc[0]*100,dev_acc[1]*100),file=logfile)

        if patience > 0 and dev_acc[0] > best_acc:
            best_acc = dev_acc[0]
            torch.save(model.state_dict(),os.path.join(args.save_dir,'model.pth'))
            patience = args.patience

        if patience == 0:
            model = SimpleQA(args).to(device)
            model.load_state_dict(torch.load(os.path.join(args.save_dir,'model.pth')))
            with torch.no_grad():
                dev_acc = model.evaluate(dev_iter)
                test_acc = model.evaluate(test_iter)
            print('Dev Acc: ({:.2f},{:.2f}), Test Acc :({:.2f},{:.2f})'.format(dev_acc[0]*100,dev_acc[1]*100,test_acc[0]*100,test_acc[1]*100))
            print('Dev Acc: ({:.2f},{:.2f}), Test Acc :({:.2f},{:.2f})'.format(dev_acc[0]*100,dev_acc[1]*100,test_acc[0]*100,test_acc[1]*100),file=logfile)
            logfile.close()
            return test_acc


def evaluate(args,test_dataset,vocab,collate_fn):

    test_iter = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=collate_fn)
    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)
    args.padding_idx = 0
    model = SimpleQA(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir,'model.pth')))
    micro_acc,macro_acc = model.evaluate(test_iter)
    return micro_acc,macro_acc


def visualize(args,labels,vocab,fname):
    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)
    args.padding_idx = 0
    model = SimpleQA(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir,'model.pth')))
    if not args.use_gcn:
        relation_embedding = model.get_relation_embedding().detach().cpu().numpy()
    else:
        relation_embedding = model.rgcn.layers[0].embedding.weight.detach().cpu().numpy()
        # relation_embedding = model.get_relation_embedding().detach().cpu().numpy()

    # normalize
    # relation_embedding -= np.mean(relation_embedding,axis=0)
    np.savetxt('embedding.tsv',relation_embedding,delimiter='\t')
    plot(relation_embedding,fname,labels)
    array = [0 for i in range(len(vocab.rtoi))]
    for rel,idx in vocab.rtoi.items():
        array[idx] = (rel,labels[idx])
    # np.savetxt('label.tsv',array,delimiter='\t')
    with open('label.tsv','w') as f:
        f.write('Relation'+'\t'+'label'+'\n')
        for rel,label in array:
            f.write(rel + '\t' + str(label) + '\n')


if __name__ == '__main__':

    args_parser = ArgumentParser()
    args_parser.add_argument('--config_file','-c',default=None,type=str)
    args_parser.add_argument('--generate',action="store_true",default=False,)
    args_parser.add_argument('--train',action="store_true",default=False)
    args_parser.add_argument('--evaluate',action="store_true",default=False)
    args_parser.add_argument('--visualize',action="store_true",default=False)
    args_parser.add_argument('--analysis',action="store_true",default=False)
    args_parser.add_argument('--self_loop',default=True,)
    args = parse_args(args_parser)
    pprint(vars(args))

    if args.generate:
        args.data_dir = os.path.join(args.data_dir,'base')
        # SimpleQADataset.generate_vocab(args)
        # SimpleQADataset.generate_embedding(args,device)
        SimpleQADataset.generate_relation_embedding(args,device)
        SimpleQADataset.generate_graph(args,device)
    elif args.train or args.evaluate or args.visualize:
        main(args)
