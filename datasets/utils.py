'''Utilities for use by modules dealing with handling data.'''

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