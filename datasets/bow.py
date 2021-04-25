'''This module contains `BOWDataset`.'''

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from datasets.vocab import Vocabulary


class BOWDataset(Dataset):
    '''This class servess as a wrapper for an existing dataset that serves
    a BOW repesentation as well as the original. The BOW conversion is done upon
    instantiation for efficiency when training.

    Attributes:
        tokenizer (:obj:`transformers.PreTrainedTokenizer`): A pretrained
            `bert-base-uncased` tokenizer from 🤗.
        vocab (:obj:`Vocabulary`): The vocabulay associated with the dataset.

    Args:
      dataset (:obj:`torch.utils.data.Dataset`): The dataset to convert to BOW.
        Expects the dataset to yield `(sentence, label)` pairs.
      vocab (:obj:`Vocabulary`): The vocabulay associated with the dataset.
      binary (:obj:`bool`, optional): Whether the BOW should maintain a binary
        representation. If `False`, it will contain word frequencies (integers).
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __init__(self, dataset, vocab, binary=False):
        self.vocab = vocab
        self.examples = []
        for sent, label in dataset:
            bow = torch.zeros((len(vocab),), dtype=torch.long)
            for word in sent.split(' '):
                word_index = vocab.words.index(word)
                if binary:
                    bow[word_index] = 1
                else:
                    bow[word_index] += 1

            self.examples.append([sent, bow, label])

    @staticmethod
    def _collate_fn(batch):
        # https://huggingface.co/transformers/preprocessing.html
        sents, bows, labels = zip(*batch)
        labels = torch.stack([x.long() for x in labels])
        encoded = BOWDataset.tokenizer(
            sents, padding=True, max_length=512, truncation=True, return_tensors='pt')
        return encoded['input_ids'], encoded['attention_mask'], torch.stack(bows).float(), labels

    def get_dataloader(self, batch_size=8, shuffle=False, num_workers=2):
        '''Handles building a PyTorch `DataLoader` for BOW-like data. The DataLoader will yield
        examples in the form `(tokenized_input, attention_mask, BOW_reprersentation, label)`. The
        input is tokenized according to this class's `tokenizer` attribute.

        Returns:
            :obj:`torch.utils.data.DataLoader`: The dataloader corresponding to this dataset.
        '''
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=BOWDataset._collate_fn)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
