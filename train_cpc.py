
from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Flatten, Dense, GRU, TimeDistributed, Input, Lambda, Dot
from keras.models import Model
from keras.optimizers import SGD, adam
from keras import backend as K

def get_encoder(x, emb_size):
    """
    Create encoder
    :param x:
    :return:
    """
    x = Conv1D(filters=10, strides=5, kernel_size=3)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=8, strides=4, kernel_size=3)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=4, strides=2, kernel_size=3)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=4, strides=2, kernel_size=3)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=4, strides=2, kernel_size=3)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(units=emb_size, activation='relu')(x)
    return x

def network_autoregressive(x, code_size):
    """
    Define the network that integrates information along the sequence
    :param x:
    :return:
    """
    return GRU(units=code_size, return_sequences=False, name='ar_context')(x)

def network_prediction(context, code_size, predict_terms):
    """
    Predict embeddings of the future steps from the context
    :param context:
    :param code_size:
    :param predict_terms:
    :return:
    """
    outputs = []
    for i in range(predict_terms):
        outputs.append(Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = Lambda(lambda x: K.stack(x, axis=1))(outputs)
    return output

def get_model():
    K.set_learning_phase(1)
    terms = 100
    predict_terms = 12
    chunk_size = [1024, 1]
    emd_size = 256

    # Define encoder model
    encoder_input = Input(shape=chunk_size)
    encoder_model = Model(encoder_input, get_encoder(encoder_input, emb_size=emd_size), name='encoder')
    #encoder_model.summary()

    # Define rest of model
    x_input = Input([terms]+chunk_size, name='Historical data')
    y_input = Input([predict_terms]+chunk_size, name='Contrastive data')

    # Workaround context
    x_encoded = TimeDistributed(encoder_model, name='Historical embeddings')(x_input)
    context = network_autoregressive(x_encoded, emd_size)

    # Make predictions for the next predict_terms timesteps
    emb_pr = network_prediction(context, emd_size, predict_terms)
    # get embeddings (it's like a ground truth)
    emb = TimeDistributed(encoder_model, name='Contrastive embeddings')(y_input)

    # Вот теперь здесь надо посчитать косунусное расстояние между эмбедингами

    def cosine_distance(vests):
        x, y = vests
        x = K.l2_normalize(x, axis=2)
        y = K.l2_normalize(y, axis=2)
        return -K.mean(x * y, axis=-1)

    def cos_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([emb_pr, emb])
    # и функцию потерь!

    # Model
    cpc_model = Model(inputs=[x_input, y_input], outputs=distance)
    cpc_model.summary()
    return cpc_model

def train():
    m = get_model()
    # Compile model
    m.compile(loss='binary_crossentropy', optimizer=adam())
    m.summary()

if __name__=='__main__':
    train()