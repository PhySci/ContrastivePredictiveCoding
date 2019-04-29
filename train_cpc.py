from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Flatten, Dense, GRU, TimeDistributed, Input, Lambda
from keras.layers import Dot, Lambda
from keras.models import Model
from keras.optimizers import SGD, adam
from keras import backend as K
import tensorflow as tf

try:
    from my_cpc.utils import ContrastiveDataGenerator, setup_logging
except ImportError:
    from ContrastivePredictiveCoding.utils import ContrastiveDataGenerator, setup_logging

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


def loss_fn(y_true, y_pred):
    """
    Contrstive loss function (eq. 4 from the original article)
    # https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras
    :param y_true: labels (0, 1), where 0 means the sample was drawn from noisy distribution; 1 means the sample was
    drawn from the target distribution.
    :param y_pred: density ratio (f value from the original article)
    :return:
    """
    divident = K.sum(K.dot(y_true, y_pred), axis=1)
    divider = K.sum(y_pred, axis=1) + K.epsilon()
    v = K.log(divident / divider)
    return -v


def get_model(chunk_size, context_samples=100, contrastive_samples=10, emd_size=512):
    """

    :param chunk_size:
    :param context_samples:
    :param contrastive_samples:
    :param emd_size:
    :return:
    """
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = Input(shape=[chunk_size, 1])
    encoder_model = Model(encoder_input, get_encoder(encoder_input, emb_size=emd_size), name='encoder')
    encoder_model.summary()

    # Define rest of the model
    x_input = Input(shape=[context_samples, chunk_size, 1], name='context_data')
    y_input = Input(shape=[contrastive_samples, chunk_size, 1], name='contrastive_data')

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
    cpc_model = Model(inputs=[x_input, y_input], outputs=f) #, y_labels
    cpc_model.summary()
    return cpc_model


def train():
    # params
    K.set_learning_phase(1)
    chunk_size = 4096
    context_samples = 5
    contrastive_samples = 1
    emd_size = 512
    batch_size = 16

    model_params = {'chunk_size': chunk_size,
                    'context_samples': context_samples,
                    'contrastive_samples': contrastive_samples,
                    'emd_size': emd_size}

    categories = ['Marimba_and_xylophone', 'Scissors', 'Gong', 'Printer', 'Keys_jangling', 'Zipper_(clothing)',
                  'Computer_keyboard', 'Finger_snapping']

    categories = ()
    gen_params = {'categories': categories,
                  'data_pth': '../data/train_curated',
                  'batch_size': batch_size,
                  'shuffle': True,
                  'seed': 42,
                  'chunk_size': chunk_size,
                  'context_samples': context_samples,
                  'contrastive_samples': contrastive_samples}


    model = get_model(**model_params)
    # Compile model
    model.compile(loss=loss_fn, optimizer=adam())

    data_gen = ContrastiveDataGenerator(**gen_params)
    model.fit_generator(generator=data_gen, epochs=10)

if __name__=='__main__':
    setup_logging('train.log')
    train()