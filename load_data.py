import numpy as np
from torchtext import data
from torchtext import datasets

""" Dataloader; Batch """
class DATA():

    def __init__(self, args, tokenizer):
        #self.TEXT = data.Field(tokenize = tokenizer, batch_first=True, lower=True, fix_length=70)
        self.TEXT = data.Field(tokenize=tokenizer, lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        if args.task_name == 'imdb':
            self.test = data.TabularDataset(
            path='./Cla_datasets/imdb/test.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

            self.train = data.TabularDataset(
            path='./Cla_datasets/imdb/train.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

            self.data = data.TabularDataset(
            path='./Example/data.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

        elif args.task_name == 'sst-2':
            self.test = data.TabularDataset(
            path='./Cla_datasets/sst-2/test.tsv', format='tsv', skip_header=True,
            fields=[('label', self.LABEL), ('sentence', self.TEXT)])

            self.train = data.TabularDataset(
            path='./Cla_datasets/sst-2/train.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

            self.data = data.TabularDataset(
            path='./Example/data.tsv', format='tsv', skip_header=True,
            fields=[('label', self.LABEL), ('sentence', self.TEXT)])

        elif args.task_name == 'cola':
            self.train = data.TabularDataset(
            path='./Cla_datasets/cola/train.tsv', format='tsv', skip_header=False,
            fields=[('sentence_source', None), ('label', self.LABEL), ('label_notes', None), ('sentence', self.TEXT)])

            self.test = data.TabularDataset(
            path='./Cla_datasets/cola/dev.tsv', format='tsv', skip_header=False,
            fields=[('sentence_source', None), ('label', self.LABEL), ('label_notes', None), ('sentence', self.TEXT)])

            self.data = data.TabularDataset(
            path='./Example/data.tsv', format='tsv', skip_header=False,
            fields=[('sentence_source', None), ('label', self.LABEL), ('label_notes', None), ('sentence', self.TEXT)])

        elif args.task_name == 'agnews':
            self.train, self.test = data.TabularDataset.splits(
            path='./Cla_datasets/'+args.task_name+'/', train='train.tsv', test='test.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

            self.data = data.TabularDataset(
            path='./Example/data.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])
        
        elif args.task_name == 'subj':
            self.train, self.test = data.TabularDataset.splits(
            path='./Cla_datasets/'+args.task_name+'/', train='train.tsv', test='test.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

            self.data = data.TabularDataset(
            path='./Example/data.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

        elif args.task_name == 'trec':
            self.train, self.test = data.TabularDataset.splits(
            path='./Cla_datasets/'+args.task_name+'/', train='train.tsv', test='test.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

            self.data = data.TabularDataset(
            path='./Example/data.tsv', format='tsv', skip_header=True,
            fields=[('sentence', self.TEXT), ('label', self.LABEL)])

        self.TEXT.build_vocab(self.train, self.test)                                
        self.LABEL.build_vocab(self.test)                                       

        self.data_iter = data.Iterator(self.data, batch_size=args.batch_size, sort=True, sort_key=lambda x: len(x.sentence), device=args.device)
