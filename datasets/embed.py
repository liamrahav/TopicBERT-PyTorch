'''This module contains `BOWDataset`.'''

import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader

from datasets.vocab import Vocabulary

class EmbeddingDataset(Dataset):
    '''This class serves as a wrapper for an existing dataset that serves
    a mapped representation of original sentences. This is useful in training
    VAE's that train their own embeddings (unlike BERT, which uses pre-trained
    embeddings & pre-trained tokenization)

    Each sentence is modified such that <sos> is the first token and <eos> is the
    last token. These correspond to embeddings `1` and `2`. <pad> is added during
    collation, and has the embedding `0`.

    Attributes:
        NUM_SPECIAL (int): The number of special tokens outside of the vocabulary
        vocab (:obj:`Vocabulary`): The vocabulay associated with the dataset.

    Args:
        dataset (:obj:`torch.utils.data.Dataset`): The dataset to convert to BOW.
            Expects the dataset to yield `(sentence, label)` pairs. Expects the the
            dataset to have attribute `labels`, an :obj:`Iterable` with the
            possible labels for any example.
        vocab (:obj:`Vocabulary`): The vocabulay associated with the dataset.
    '''
    NUM_SPECIAL = 3

    def __init__(self, dataset, vocab):
        self.vocab = vocab
        self.examples = []
        for sent, *_ in dataset:
            # add 3 to account for our special tokens
            sent = [1] + [vocab.words.index(word) + self.NUM_SPECIAL for word in sent.split(' ')] + [2]
            self.examples.append(sent)


    def decode_sentence(self, sent):
        '''Decodes a sentece represented by integer token ids'''
        ret = ''
        for token in sent:
            if token == 0:
                ret += '<pad>'
            elif token == 1:
                ret += '<sos>'
            elif token == 2:
                ret += '<eos>'
            else:
                ret += self.vocab.words[token - self.NUM_SPECIAL]
            ret += ' '
        return ret


    @staticmethod
    def _collate_fn(batch):
        batch = [torch.LongTensor(sent) for sent in batch]
        return rnn.pad_sequence(batch)


    def get_dataloader(self, batch_size=8, shuffle=False, num_workers=2):
        '''Handles building a PyTorch `DataLoader` for BOW-like data. The DataLoader will yield
        examples in the form `(tokenized_input, attention_mask, BOW_reprersentation, label)`. The
        input is tokenized according to this class's `tokenizer` attribute.

        Returns:
            :obj:`torch.utils.data.DataLoader`: The dataloader corresponding to this dataset.
        '''
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=EmbeddingDataset._collate_fn)

    @property
    def num_labels(self):
        '''int: useful when instantiating PyTorch modules.'''
        return len(self.labels)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
