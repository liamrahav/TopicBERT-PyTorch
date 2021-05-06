import os
import pickle

import torch
from torch.utils.data import Dataset
from torchtext.datasets import IMDB
from torchtext.data import RawField
from tqdm import tqdm

from datasets import Vocabulary
from datasets.utils import TwoWayDict


class IMDBDataset(Dataset):
    def __init__(self, examples, vocab, one_hot=True, silent=False):
        # Remove words not in the vocab from examples
        self.examples = []
        words = vocab.words
        for sent, label in tqdm(examples, total=len(examples), disable=silent):
            sent = ' '.join([x for x in sent.lower().split(' ') if x in words])
            self.examples.append([sent, label])

        self.vocab = vocab
        self.labels = ['neg', 'pos']  # So that neg == 0
        self.label_mapping = TwoWayDict()
        for i, label in enumerate(self.labels):
            self.label_mapping[label] = i

        if one_hot:
            for i, ex in enumerate(self.examples):
                _, label = ex
                lbl_one_hot = torch.full((len(self.labels),), 0)
                lbl_one_hot[self.label_mapping[label]] = 1
                self.examples[i][1] = lbl_one_hot

    @classmethod
    def full_split(cls, root_dir, val_size=1000, load_processed=True, save_processed=True):
        '''Generates the full train/val/test split'''
        spd = os.path.join(root_dir, 'imdb', 'processed/')
        train_path = os.path.join(spd, 'train.pkl')
        val_path = os.path.join(spd, 'val.pkl')
        test_path = os.path.join(spd, 'test.pkl')
        if (load_processed and os.path.exists(train_path) and os.path.exists(val_path)
            and os.path.exists(test_path)):
            print(" [*] Loading pre-processed IMDB objects.")
            with open(train_path, 'rb') as train_f, open(val_path, 'rb') as val_f, open(test_path, 'rb') as test_f:
                return pickle.load(train_f), pickle.load(val_f), pickle.load(test_f)

        # This means we're not loading from pickle
        itrain, itest = IMDB.splits(RawField(), RawField(), root=root_dir)

        vocab = Vocabulary([x.text for x in itrain] +
                           [x.text for x in itest], f_min=100)

        # For val we take middle val_size values as this is where pos/neg switch occurs
        mid = len(itrain) // 2
        grab = val_size // 2
        train = cls([[x.text, x.label] for x in itrain[:mid - grab]] +
                    [[x.text, x.label] for x in itrain[mid + grab:]], vocab)
        val = cls([[x.text, x.label]
                   for x in itrain[mid - grab:mid + grab]], vocab)
        test = cls([[x.text, x.label] for x in itest], vocab)

        if save_processed:
            if not os.path.exists(spd):
                os.makedirs(spd)

            with open(train_path, 'wb') as f:
                pickle.dump(train, f)

            with open(val_path, 'wb') as f:
                pickle.dump(val, f)

            with open(test_path, 'wb') as f:
                pickle.dump(test, f)

        return train, val, test

    @property
    def num_labels(self):
        '''int: useful when instantiating PyTorch modules.'''
        return len(self.labels)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
