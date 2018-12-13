import spacy
import os
import numpy as np

class Rambler(object):

    def __init__(self, model, nlp, indexer):
        self.model = model
        self.indexer = indexer
        self.nlp = nlp

    def __call__(self, string):
        doc = self.nlp(string)
        seq_len = self.model.input_shape[1]
        doc_len = len(doc)

        x = np.zeros((1, self.model.input_shape[1], self.model.input_shape[2]))

        doc_start = max(0, len(doc) - seq_len)
        x_start = max(0, seq_len - len(doc))

        for i, token in enumerate(doc[doc_start:doc_len]):
            x[0, i+x_start, :] = token.vector

        predictions = self.model.predict(x)
        indices = map(lambda p: np.random.choice(p.shape[1], p=p[0,:]),
                predictions)
        indices = tuple(indices)

        new_str = self.indexer(indices)
        return new_str, string + new_str


def nlp_doc(sample = 1e6):
    paths = ["../data/rowling.txt",
            "./data/rowling.txt"
            "/floyd/input/rowling/rowling.txt"]

    path = filter(os.path.isfile, paths)[0]

    txt = open(path, "r").read()
    txt = unicode(txt, "UTF-8")

    if sample is not None:
        txt = txt[0:int(sample)]

    nlp = spacy.load("en_core_web_md")
    nlp.max_length = 2*len(txt)

    doc = nlp(txt, disable=['tagger', 'parser', 'ner'])

    return nlp, doc
