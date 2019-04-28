from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Flatten, Dense, GRU, TimeDistributed, Input, Lambda
from keras.layers import Dot, Lambda
from keras.models import Model
from keras.optimizers import SGD, adam
from keras import backend as K
import tensorflow as tf

def get_encoder(x, emb_size):
    """
    Create encoder
    :param x:
    :return:
    """
    with tf.name_scope('embedding_level_1'):
        x = Conv1D(filters=10, strides=5, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

    with tf.name_scope('embedding_level_2'):
        x = Conv1D(filters=8, strides=4, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

    with tf.name_scope('embedding_level_2'):
        x = Conv1D(filters=4, strides=2, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

    with tf.name_scope('embedding_level_4'):
        x = Conv1D(filters=4, strides=2, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

    with tf.name_scope('embedding_level_5'):
        x = Conv1D(filters=4, strides=2, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

    with tf.name_scope('embedding_dense'):
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


def loss_fn(y_true, y_pred):
    """
    Contrstive loss function (eq. 4 from the original article)
    # https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras
    :param y_true: labels (0, 1), where 0 means the sample was drawn from noisy distribution; 1 means the sample was drawn
    from the target distribution.
    :param y_pred: density ratio (f value from the original article)
    :return:
    """
    divident = K.sum(K.dot(y_true, y_pred), axis=1)
    divider = K.sum(y_pred, axis=1)
    v = K.log(divident / divider)
    return K.mean(v)


def get_model():
    K.set_learning_phase(1)
    terms = 100
    predict_terms = 12
    n_samples = 10
    chunk_size = [1024, 1]
    emd_size = 512

    # Define encoder model
    encoder_input = Input(shape=chunk_size)
    encoder_model = Model(encoder_input, get_encoder(encoder_input, emb_size=emd_size), name='encoder')
    encoder_model.summary()

    # Define rest of the model
    x_input = Input([terms]+chunk_size, name='context_data')
    y_input = Input([n_samples]+chunk_size, name='contrastive_data')
    y_labels = Input([n_samples], name='labels')

    # Workaround context
    x_encoded = TimeDistributed(encoder_model, name='Historical_embeddings')(x_input)
    context = network_autoregressive(x_encoded, emd_size)

    # Make predictions for the next predict_terms timesteps
    z = TimeDistributed(encoder_model, name='Contrastive_embeddings')(y_input)
    # Equation 3
    z1 = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='transpose')(z)
    z2 = Dense(units=emd_size, name='W')(context)
    f = Lambda(lambda x: K.exp(Dot(axes=1)(x)), name='multiplication')([z1, z2])

    # Model
    cpc_model = Model(inputs=[x_input, y_input, y_labels], outputs=f)
    cpc_model.summary()
    return cpc_model

def train():
    m = get_model()
    # Compile model
    m.compile(loss=loss_fn, optimizer=adam())

if __name__=='__main__':
    train()