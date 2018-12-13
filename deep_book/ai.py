import tensorflow.keras as ks

def lstm(indexer, n_layers=2, n_hidden=32, seq_len=64):

    input_vector = ks.Input((seq_len,300))

    x = ks.layers.LSTM(n_hidden, return_sequences=True)(input_vector)

    for i in range(1, n_layers):
        rs = i != (n_layers-1)
        x = ks.layers.LSTM(n_hidden, return_sequences=rs)(x)

    norm_pred = ks.layers.Dense(indexer.n_norm, activation="softmax")(x)
    shape_pred = ks.layers.Dense(indexer.n_shape, activation="softmax")(x)
    whitespace_pred = ks.layers.Dense(indexer.n_whitespace,
            activation="softmax")(x)

    model = ks.Model(input_vector, [norm_pred, shape_pred, whitespace_pred])

    return model

def gpu_lstm(indexer, n_lstm=2, n_hidden=32, seq_len=64):
    input_sequence = ks.Input(shape = (sequence_length, 300))

    # project initial word vectors onto same dimensions as LSTM
    x = ks.layers.Dense(n_hidden_lstm)(input_sequence)

    # add LSTMs with residual connections
    for i in range(n_lstm - 1):
        resid = ks.layers.CuDNNLSTM(n_hidden_lstm,
                return_sequences=True, name = "lstm_%i" % (i))(x)
        x = ks.layers.Add()([x, resid])

    resid = ks.layers.CuDNNLSTM(n_hidden_lstm,
        return_sequences=False, name = "lstm_%i" % (n_lstm-1))(x)
    x = ks.layers.Lambda(lambda x: x[:, n_hidden_lstm - 1, :])(x)
    x = ks.layers.Add()([x, resid])

    norm_pred = ks.layers.Dense(indexer.n_norm, activation="softmax")(x)
    shape_pred = ks.layers.Dense(indexer.n_shape, activation="softmax")(x)
    whitespace_pred = ks.layers.Dense(indexer.n_whitespace,
            activation="softmax")(x)

    model = ks.Model(input_vector, [norm_pred, shape_pred, whitespace_pred])

    return model
