import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.entity_typing_dataloader import EntityTypingDataset
from learner.entity_typing import Model

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
    vocab = EntityTypingDataset.load_vocab(args)
    train_dataset,dev_dataset,test_dataset = EntityTypingDataset.load_dataset(args)

    end_time = time.time()
    print('Loaded dataset in {:.2f}s '.format(end_time - start_time))

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

            test_acc = train(args,train_subset,dev_subset,test_subset,vocab,EntityTypingDataset.collate_fn)
            with open('fold/fold{}.txt'.format(str(i)),'a') as f:
                f.write('Test Acc :({:.2f},{:.2f},{:.2f}'.format(test_acc[0],test_acc[1],test_acc[2]))

    elif args.mode == 'supervised':
        train(args,train_dataset,dev_dataset,test_dataset,vocab,EntityTypingDataset.collate_fn)


def train(args,train_dataset,dev_dataset,test_dataset,vocab,collate_fn):

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
        EntityTypingDataset.generate_dataset(args)
        EntityTypingDataset.generate_embedding(args,device)
