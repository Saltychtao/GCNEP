import torch
import os
import numpy as np
import json
from pprint import pprint
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from dataloader.simpleQA_dataloader import SimpleQADataset
from model.SimpleQA import SimpleQA
from utils.graph_util import build_graph_from_adj_matrix,get_seen_density
from utils.visualize import plot_embedding,plot_density
from utils.util import parse_args,pairwise_distances

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
        args.adj_matrix = []
        for pth in args.relation_adj_matrix_pth:
            adj_matrix = torch.load(pth)
            print('Relation Adj matrix loaded!')
            print('Building Relation Graph ...')
            if not args.self_loop:
                # remove self loop
                print('Removing Self-Loop')
                for i in range(adj_matrix.shape[0]):
                    adj_matrix[i][i] = 0
            g = build_graph_from_adj_matrix(adj_matrix,device,args.norm_type)
            print('Done.')
            args.adj_matrix.append(adj_matrix)
            args.relation_graphs.append(g)
        if args.graph_aggr == 'concat':
            args.sub_relation_dim = args.relation_dim // len(args.relation_graphs)
        else:
            args.sub_relation_dim = args.relation_dim

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
    if args.dataset == 'base':
        train_fname = os.path.join(base_data_dir,'base','train.tsv')
        dev_fname = os.path.join(base_data_dir,'base','dev.tsv')
        test_fname = os.path.join(base_data_dir,'base','test.tsv')
        train_dataset,dev_dataset,test_dataset = SimpleQADataset.load_dataset(train_fname,dev_fname,test_fname,args.vocab_pth,args)
        if args.train:
            print(' Training On origin dataset...')
            train(args,train_dataset,dev_dataset,test_dataset,vocab,SimpleQADataset.collate_fn)
        elif args.evaluate:
            test_micro_acc,test_macro_acc = evaluate(args,test_dataset,vocab,SimpleQADataset.collate_fn)
            print(' Test Acc on origin dataset: {:.2f},{:.2f}'.format(test_micro_acc,test_macro_acc))

    elif args.dataset == 'mix':
        for i in range(args.fold):
            train_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'train.tsv')
            dev_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'dev.tsv')
            test_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'test.tsv')
            test_seen_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'test_seen.tsv')
            test_unseen_fname = os.path.join(base_data_dir,'fold-{}'.format(i),'test_unseen.tsv')
            train_dataset,dev_dataset,test_dataset,test_seen_dataset,test_unseen_dataset = SimpleQADataset.load_dataset([train_fname,dev_fname,test_fname,test_seen_fname,test_unseen_fname],args.vocab_pth,args)
            args.save_dir = os.path.join(base_save_dir,'fold-{}'.format(str(i)))

            train_relations = train_dataset.get_label_set()
            args.seen_idx = list(train_relations)
            args.unseen_idx = list(set([i for i in range(len(vocab.rtoi))]) - set(args.seen_idx))
            label_idx = np.zeros(len(vocab.rtoi))
            label_idx[args.seen_idx] = 1
            args.label_idx = label_idx

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
                    seen_micro_acc,seen_macro_acc = evaluate(args,test_seen_dataset,vocab,SimpleQADataset.collate_fn)
                    print('Seen Acc :({:.2f},{:.2f})'.format(seen_micro_acc*100,seen_macro_acc*100))

                    # Unseen
                    unseen_micro_acc,unseen_macro_acc = evaluate(args,test_unseen_dataset,vocab,SimpleQADataset.collate_fn)
                    print('UnSeen Acc :({:.2f},{:.2f})'.format(unseen_micro_acc*100,unseen_macro_acc*100))
            elif args.visualize:
                print(' Visualizing Fold {}'.format(i))
                fname = os.path.join(args.save_dir,'embedding.png')
                visualize(args,label_idx,vocab,fname)
            elif args.analysis:
                print(' Analyzing Fold {}'.format(i))
                first_order_fname = os.path.join(args.save_dir,'first_order.png')
                second_order_fname = os.path.join(args.save_dir,'second_order.png')
                analysis(args,vocab,test_dataset,SimpleQADataset.collate_fn,first_order_fname,second_order_fname)


def train(args,train_dataset,dev_dataset,test_dataset,vocab,collate_fn):

    train_iter = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=12,collate_fn=collate_fn)
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
        relation_embedding = model.get_relation_embedding().detach().cpu().numpy()

    # normalize
    # relation_embedding -= np.mean(relation_embedding,axis=0)
    np.savetxt('embedding.tsv',relation_embedding,delimiter='\t')
    plot_embedding(relation_embedding,fname,labels)
    array = [0 for i in range(len(vocab.rtoi))]
    for rel,idx in vocab.rtoi.items():
        array[idx] = (rel,labels[idx])
    # np.savetxt('label.tsv',array,delimiter='\t')
    with open('label.tsv','w') as f:
        f.write('Relation'+'\t'+'label'+'\n')
        for rel,label in array:
            f.write(rel + '\t' + str(label) + '\n')


def analysis(args,vocab,test_dataset,collate_fn,first_order_fname,second_order_fname):
    # Load Model
    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)
    args.padding_idx = 0
    model = SimpleQA(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir,'model.pth')))

    # Load Graph
    adj_matrix = torch.load(args.relation_adj_matrix_pth[0])

    # Load Dataset
    test_iter = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=12,collate_fn=collate_fn)
    with torch.no_grad():
        gold,pred,k_preds,ranks = model.predict(test_iter)
        # if not args.use_gcn:
        #     relation_embedding = model.get_relation_embedding()
        # else:
        #     relation_embedding = model.get_relation_embedding()
        # pdist = pairwise_distances(relation_embedding)

    # first_order_seen_density = get_seen_density(adj_matrix,args.seen_idx,args.unseen_idx,order=1)
    # second_order_seen_density = get_seen_density(adj_matrix,args.seen_idx,args.unseen_idx,order=2)
    # plot_density(pred,gold,first_order_seen_density,args.seen_idx,first_order_fname)
    # plot_density(pred,gold,second_order_seen_density,args.seen_idx,second_order_fname)

    with open(os.path.join(args.save_dir,'test.errors'),'w') as f:
        outputs = []
        for i,(p,g,k,r) in enumerate(zip(pred,gold,k_preds,ranks)):
            if p != g:
                raw_line = test_dataset.get_raw_instance(i)
                gold,neg,question = raw_line.rstrip().split('\t')
                gold_relation = vocab.itor[g]
                neg_relations = []
                for n in neg.split():
                    try:
                        idx = int(n)
                        neg_relations.append(vocab.itor[idx])
                    except ValueError:
                        pass
                d = dict()
                d['question'] = question
                d['gold'] = gold_relation
                d['gold_type'] = 'seen' if g in args.seen_idx else 'unseen'
                d['pred'] = vocab.itor[p]
                d['pred_type'] = 'seen' if p in args.seen_idx else 'unseen'
                d['negative'] = ' '.join(neg_relations)
                d['gold_neighbour_seen'] = (adj_matrix[g] * args.label_idx).sum()
                d['gold_neighbour_unseen'] = (adj_matrix[g] * (1-args.label_idx)).sum()
                d['gold_total_neighbour'] = adj_matrix[g].sum()
                d['pred_neighbour_seen'] = (adj_matrix[p] * args.label_idx).sum()
                d['pred_neighbour_unseen'] = (adj_matrix[p] * (1-args.label_idx)).sum()
                d['pred_total_neighbour'] = adj_matrix[p].sum()
                d['rank'] = r
                d['top_five_relations'] = []
                for j in k:
                    t = (vocab.itor[j],'seen' if j in args.seen_idx else 'unseen',adj_matrix[int(j)].sum())
                    d['top_five_relations'].append(t)
                outputs.append(d)
        json.dump(outputs,f,indent=4)


if __name__ == '__main__':

    args_parser = ArgumentParser()
    args_parser.add_argument('--config_file','-c',default=None,type=str)
    args_parser.add_argument('--generate',action="store_true",default=False,)
    args_parser.add_argument('--train',action="store_true",default=False)
    args_parser.add_argument('--evaluate',action="store_true",default=False)
    args_parser.add_argument('--visualize',action="store_true",default=False)
    args_parser.add_argument('--analysis',action="store_true",default=False)
    args_parser.add_argument('--graph_aggr',type=str,default='concat')
    args_parser.add_argument('--self_loop',default=False,)
    args_parser.add_argument('--dataset',default='mix')
    args_parser.add_argument('--norm_type',default='spectral')
    args = parse_args(args_parser)
    pprint(vars(args))

    if args.generate:
        args.data_dir = os.path.join(args.data_dir,'base')
        # SimpleQADataset.generate_vocab(args)
        # SimpleQADataset.generate_embedding(args,device)
        SimpleQADataset.generate_relation_embedding(args,device)
        SimpleQADataset.generate_graph(args,device)
    elif args.train or args.evaluate or args.visualize or args.analysis:
        if args.visualize or args.analysis:
            args.fold = 10
        main(args)
