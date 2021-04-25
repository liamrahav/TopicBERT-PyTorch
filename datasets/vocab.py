'''This  module contains the `Vocabulary` class.'''
from collections import Counter

import nltk
from nltk.corpus import stopwords

from datasets.utils import read_tsv


class Vocabulary():
    '''Creates a dictionary-like vocabulary from a raw, tab-separated dataset.

    `Vocabulary` is essentially a wrapper on the builtin `Counter` class. This class
    implements __getitem__, which returns the frequency of a given word.

    Args:
        filenames (:obj:`list` of :obj:`str`): Names of TSV files making up the full
            dataset whose vocabulary you want to index. Typically 'train', 'val', and
            'test' files.
        col_num (:obj:`int`, optional): Set to `1` by default. The column in the TSV
            containing raw text. This starts from 0 (e.g. 1 is the 2nd column).
        lowercase (:obj:`bool`, optional): Set to `True` by default. If `True`, will
            lowercase all words in the corpus.
        include_stopwords (:obj:`bool`, optional): Set to `False` by default. If `True`,
            the vocabulary will include words listed in NLTK's stopwords dataset. Upon
            initialization, stopwords will be downloaded if needed.
        f_min(:obj:`int`, optional): Set to `10` by default. All words with
            frequency < f_min are removed from the vocabulary.

    '''

    def __init__(self, filepaths, col_num=1, lowercase=True, include_stopwords=False, f_min=10):
        texts = []
        if not filepaths:
            raise ValueError('filepaths must contain at least one file')

        for filepath in filepaths:
            texts += [x[col_num] for x in read_tsv(filepath)]

        words = [word.lower() for text in texts for word in text.split(' ')]

        if not include_stopwords:
            # remove stopwords, newline
            nltk.download('stopwords', quiet=True)
            stops = stopwords.words('english')
            words = [x for x in words if x not in stops]

        c = Counter(words)
        self._vocab = Counter({k: c for k, c in c.items() if c >= f_min})

    @property
    def words(self):
        ''':obj:`list` of :obj:`str`: All words in the vocabulary'''
        return list(self._vocab.keys())

    def __getitem__(self, word):
        return self._vocab[word]

    def __len__(self):
        return len(self.words)
