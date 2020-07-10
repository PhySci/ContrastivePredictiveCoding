import tensorflow as tf
from utils import ContrastiveDataGenerator

def train():
    pass

def get_embedding_layer(x, layer_name, conv_params={}):
    with tf.name_scope(layer_name):
        y = tf.layers.conv1d(x, **conv_params)
        y = tf.layers.BatchNormalization(y)
    return y

def get_encoder(x, emb_size):
    """
    Create encoder
    :param x:
    :return:
    """
    with tf.name_scope('Encoder'):

        conv_params1 = {'filters': 10,
                       'kernel_size': 3,
                       'strides': 5,
                       'activation': 'relu'}
        y = get_embedding_layer(x, layer_name='embedding_level_1', conv_params=conv_params1)

        conv_params2 = {'filters': 8,
                       'kernel_size': 3,
                       'strides': 4,
                       'activation': 'relu'}
        y = get_embedding_layer(y, layer_name='embedding_level_2', conv_params=conv_params2)

        conv_params3 = {'filters': 4,
                       'kernel_size': 3,
                       'strides': 2,
                       'activation': 'relu'}
        y = get_embedding_layer(y, layer_name='embedding_level_3', conv_params=conv_params3)

        conv_params4 = {'filters': 4,
                       'kernel_size': 3,
                       'strides': 2,
                       'activation': 'relu'}
        y = get_embedding_layer(y, layer_name='embedding_level_3', conv_params=conv_params4)

        conv_params5 = {'filters': 4,
                       'kernel_size': 3,
                       'strides': 2,
                       'activation': 'relu'}
        y = get_embedding_layer(y, layer_name='embedding_level_3', conv_params=conv_params5)

        with tf.name_scope('embedding_dense'):
            y = tf.layers.flatten(y)
            y = tf.layers.dense(y, units=emb_size, activation='relu')
    return y

def get_context():
    encoder = get_encoder()
    pass

def get_model(chunk_size, context_samples=100, contrastive_samples=10, emd_size=512, gru_size=256):
    """

    :param chunk_size:
    :param context_samples:
    :param contrastive_samples:
    :param emd_size:
    :param gru_size:
    :return:
    """

def get_context(context_tensor):
    """

    :param context_tensor: tensor [batch_size, n_context_samples, chunk_size, 1]
    :return:
    """

    pass

def build_graph():
    pass

def main():

    chunk_size = 4096
    context_samples = 5
    contrastive_samples = 1
    emd_size = 512
    batch_size = 8

    categories = ['Marimba_and_xylophone', 'Scissors', 'Gong', 'Printer', 'Keys_jangling', 'Zipper_(clothing)', 'Computer_keyboard', 'Finger_snapping']

    gen_params = {'categories': categories,
                  'data_pth': '../data/train_curated',
                  'batch_size': batch_size,
                  'shuffle': True,
                  'seed': 42,
                  'chunk_size': chunk_size,
                  'context_samples': context_samples,
                  'contrastive_samples': contrastive_samples}

    data_gen = ContrastiveDataGenerator(**gen_params)

    with tf.Session() as sess:
        for (context_batch, contrastive_batch), labels in data_gen:
            print(context_batch.shape)
            #№c = get_encoder(None, emd_size)


if __name__ == '__main__':
    main()