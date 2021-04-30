'''Utilities for use by modules dealing with handling data.'''
from torch.utils.data import Dataset


class PartitionedDataset(Dataset):
    '''This class serves as a wrapper for an existing dataset that splits
    examples in the original dataset up to some partition factor. It truncates
    sentences from right to left as to avoid creating unnecessary padding.

    Args:
        dataset (:obj:`torch.utils.data.Dataset`): The dataset to convert to BOW.
            Expects the dataset to yield `(sentence, label)` pairs.
        max_length (:obj:`int`, optional): Defaults to :obj:`512`. Sets how long
            a sentence may be. Partitions are calculated with respect to this.
        partition_factor (:obj:`int`, optional): Defaults to :obj:`1`. Controls how
            long examples can be relative to :obj:`max_length`. E.g. if :obj:`
            partition_factor = 4`, :obj:`BOWDataset` will split examples such that
            sentences have a max length of :math:`512 / 4 = 128`. It is recommended
            that this be a power of 2.
    '''
    def __init__(self, dataset, max_length=512, partition_factor=1):
        self.max_length = max_length
        self.partition_factor = partition_factor
        sent_len = self.max_length // self.partition_factor

        self.examples = []
        for sent, label in dataset:
            if len(sent) > self.max_length:
                sent = sent[:max_length]

            while len(sent) > sent_len:
                self.examples.append([sent[-sent_len:], label])
                sent = sent[:-sent_len]

            self.examples.append([sent, label])

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


class TwoWayDict(dict):
    '''Thin wrapper on  `dict` that enforces two-way-ness.
    Taken from https://stackoverflow.com/questions/1456373/two-way-reverse-map.

    Example:
        If `twd` is a `TwoWayDict`, then

        >>> twd[a] = b
        >>> twd[b]
        a
    '''
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2

def read_tsv(path):
    '''Converts a tab-separated file to Python representation.

    Args:
        path (str): The full path to the tab-separated file.

    Returns:
        :obj:`list` of :obj:`list` of :obj:`str`: The parsed data.
    '''
    results = []
    with open(path) as f:
        f.readline()
        for line in f:
            results.append(line.split('\t'))
    return results