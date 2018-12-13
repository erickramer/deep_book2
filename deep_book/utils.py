import tensorflow
import tensorflow.keras as ks
import os

class RamblerCallback(ks.callbacks.Callback):

    def __init__(self, rambler, path = "./rambles", n_rambles=5, ramble_len=128):
        super(RamblerCallback, self).__init__()
        self.rambler = rambler
        self.path = path
        self.n_rambles = n_rambles
        self.ramble_len = ramble_len

        if not os.path.isdir(path):
            os.mkdir(path)

    def on_epoch_end(self, epoch, logs):
        with open("./rambles/rambles_%i.txt" % epoch, "w") as o:
            for _i in range(self.n_rambles):
                string = u''
                for _j in range(self.ramble_len):
                    token, string = self.rambler(string)
                o.write(string.encode('utf-8'))
                o.write("\n\n")
