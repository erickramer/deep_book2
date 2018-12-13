import spacy
import numpy as np

class Indexer(object):

    def __init__(self, doc, min_occurences = 10):
        self._doc = doc
        self._min_occurances = min_occurences

        norms = {}
        shapes = {}
        whitespaces = {}
        for token in doc:
            if token.norm_ not in norms:
                norms[token.norm_] = 1
            else:
                norms[token.norm_] += 1

            if token.shape_ not in shapes:
                shapes[token.shape_] = 1
            else:
                shapes[token.shape_] += 1

            if token.whitespace_ not in whitespaces:
                whitespaces[token.whitespace_] = 1
            else:
                whitespaces[token.whitespace_] += 1

        norms_reduced = {norm: count \
            for norm, count in norms.iteritems() if count > 10}
        shapes_reduced = {shape: count \
            for shape, count in shapes.iteritems() if count > 10}
        whitespaces_reduced = {ws: count \
            for ws, count in whitespaces.iteritems() if count > 10}

        self.n_norm = len(norms_reduced)
        self.n_shape = len(shapes_reduced)
        self.n_whitespace = len(whitespaces_reduced)

        self._norm2row = {norm: i \
            for i, norm in enumerate(norms_reduced.keys())}
        self._shape2row = {shape: i \
            for i, shape in enumerate(shapes_reduced.keys())}
        self._whitespace2row = {ws: i \
            for i, ws in enumerate(whitespaces_reduced.keys())}

        self._row2norm = {i: norm \
            for i, norm in enumerate(norms_reduced.keys())}
        self._row2shape = {i: shape \
            for i, shape in enumerate(shapes_reduced.keys())}
        self._row2whitespace = {i: ws \
            for i, ws in enumerate(whitespaces_reduced.keys())}

    def __call__(self, x):
        if type(x) == tuple:
            norm = self.norm(x[0])
            shape = self.shape(x[1])
            whitespace = self.whitespace(x[2])
            return norm + whitespace
        else:
            return (self.norm(x.norm_),
                    self.shape(x.shape_),
                    self.whitespace(x.whitespace_))

    def _lookup(self, x, int2str, str2int):
        if type(x) == int:
            if x in int2str:
                return int2str[x]
            else:
                return None
        else:
            if x in str2int:
                return str2int[x]
            else:
                return None
    def norm(self, x):
        return self._lookup(x, self._row2norm, self._norm2row)

    def shape(self, x):
        return self._lookup(x, self._row2shape, self._shape2row)

    def whitespace(self, x):
        return self._lookup(x, self._row2whitespace, self._whitespace2row)

    def generator(self, seq_length=64, batch_size=100, randomize=True):
        n_tokens = len(self._doc)

        # initializing arrays
        xs = []
        targets = []

        inds = range(0, n_tokens - seq_length)

        while True:
            if randomize:
                np.random.shuffle(inds)

            for i in inds:

                target = self._doc[i+seq_length]
                target = self(target)

                if all(map(lambda x: x is not None, target)):
                    x = np.stack([token.vector \
                                    for token in self._doc[i:(i+seq_length)]])

                    xs.append(x)
                    targets.append(target)

                if len(xs) == batch_size:
                    yield np.stack(xs), map(np.stack, zip(*targets))
                    xs = []
                    targets = []
