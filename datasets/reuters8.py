'''This module contains the `Reuters8Dataset` class'''

import torch
from torch.utils.data import Dataset

from datasets.vocab import Vocabulary
from datasets.utils import read_tsv, TwoWayDict


class Reuters8Dataset(Dataset):
    '''Initializes a PyTorch Dataset with the appropriate Reuters8 data.

    Args:
        filepath (str): The path of the file containing Ruters8 label/sentence pairs as TSV.
        label_filepath (str): The label file containing the label ordering, separated by
            newlines.
        vocab (:obj:`Vocabulary`): A vocabulary object that determines which words are retained by
            the dataset when cleaning examples.
        one_hot (:obj:`bool`, optional): Set to True by default. Determines if labels are one-hot encoded
            vectors (True) or topic strings (False)
    '''

    def _clean_examples(self, examples):
        # Switches order of label, sent in dataset. Removes words not in vocab.
        cleaned = []
        words = self.vocab.words
        for ex in examples:
            label, sent = ex
            new_sent = ' '.join([x.lower()
                                for x in sent.split(' ') if x.lower() in words])
            cleaned.append([new_sent, label])

        return cleaned

    def __init__(self, filepath, label_filepath, vocab, one_hot=True):
        self.vocab = vocab

        with open(label_filepath) as f:
            self.labels = [x.strip() for x in f.read().strip().split('\n')]

        self.label_mapping = TwoWayDict()
        for i, label in enumerate(self.labels):
            self.label_mapping[label] = i

        self.examples = self._clean_examples(read_tsv(filepath))

        if one_hot:
            for i, ex in enumerate(self.examples):
                _, label = ex
                lbl_one_hot = torch.full((len(self.labels),), 0)
                lbl_one_hot[self.label_mapping[label]] = 1
                self.examples[i][1] = lbl_one_hot

    @property
    def num_labels(self):
        '''int: useful when instantiating PyTorch modules.'''
        return len(self.labels)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
