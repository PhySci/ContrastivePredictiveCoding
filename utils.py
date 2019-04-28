from keras.utils import Sequence
from random import shuffle
import pandas as pd
import numpy as np
import librosa
import os
import logging
import warnings
from random import shuffle

class SignalGenerator(Sequence):

    def __init__(self, data_file='../input/train.parquet', meta_file='../input/metadata_train.csv',
                 batch_size=10, shuffle=True, seed=42, test_mode=False, measurement_ids=None,
                 normalize=True, compress_rate=1,
                 wavelets=0, wavelet_params={}):
        """
        Constructor

        :param data_file: path to data file
        :param meta_file:  path to meta file
        :param batch_size: batch size
        :param measurement_ids: list of measurement ids. Dedicated for CV
        :param shuffle:
        :param seed: random seed
        :param test_mode: return samples and signal_ids
        :param normalize: to normalize the data
        """
        self.it = 0
        self.shuffle = shuffle
        self.data_pth = data_file
        self.batch_size = batch_size
        self.seed = seed
        self.test_mode = test_mode

        meta_df = pd.read_csv(meta_file)
        if measurement_ids is not None:
            meta_df = meta_df.query('id_measurement in @measurement_ids')

        self.meta_df = meta_df.set_index('id_measurement', drop=True)
        self.measurement_id_list = np.unique(self.meta_df.index.values)
        self.list_sz = len(self.measurement_id_list)
        self.max_it = int(np.ceil(self.list_sz / self.batch_size))
        self.normalize = normalize
        self.wavelets = wavelets
        self.wavelet_params = wavelet_params

        if 800000 % compress_rate !=0:
            raise ValueError('Compress rate is not integer divider of signal length')
        self.compress_rate = compress_rate

    def __len__(self):
        return self.max_it

    def on_epoch_end(self):
        """
        Performs at the end of each epoch
        :return:
        """
        shuffle(self.measurement_id_list)

    def __getitem__(self, item):
        """
        Return one batch
        :param item:
        :return:
        """
        return self.__data_generation(item)

    def __data_generation(self, it):
        """
        Data generator
        :param it:
        :return:
        """
        start = np.minimum(it * self.batch_size, self.list_sz)
        end = np.minimum(start + self.batch_size, self.list_sz)
        # list of measurements ids
        measurement_id_batch = self.measurement_id_list[start:end]
        # get list of meta info for the given measurements ids
        data_batch = self.meta_df.loc[measurement_id_batch]

        cols = data_batch.signal_id.astype(str).tolist()
        data = pq.read_pandas(self.data_pth, columns=cols).to_pandas().values

        if self.compress_rate > 1:
            data = data.reshape([self.compress_rate, data.shape[0]//self.compress_rate, 3, -1], order='F')
            data = data.sum(axis=0)
        else:
            data = data.reshape([data.shape[0], 3, -1], order='F')
        data = np.moveaxis(data, -1, 0)

        labels = data_batch.target.values.reshape([-1, 3])

        if self.normalize:
            data = data - data.mean(axis=(1, 2), keepdims=True)
            data = data / data.std(axis=(1, 2), keepdims=True)

        if self.wavelets == 1:
            data = pywt.wavedec(data, axis=1, **self.wavelet_params)
        elif self.wavelets ==2:
            r = pywt.swt(data, axis=1, **self.wavelet_params)
            ca = np.zeros([data.shape[0], data.shape[1], self.wavelet_params.get('level'), 3])
            cd = np.zeros([data.shape[0], data.shape[1], self.wavelet_params.get('level'), 3])
            for i, el in enumerate(r):
                #ca[:, :, i, :] = el[0]
                cd[:, :, i, :] = el[1]

            #data = ca.reshape([data.shape[0], data.shape[1], -1], order='F')
            data = cd.reshape([data.shape[0], data.shape[1], -1], order='F')

        if self.test_mode:
            return data, data_batch.signal_id
        else:
            return data, [labels[:, 0], labels[:, 1], labels[:, 2]]

def setup_logging(fname, level=logging.DEBUG):
    """
    Create logger instance
    :param fname: name of log file
    :param level: log level
    :return:
    """
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    logger = logging.getLogger()
    logger.setLevel(level)

    # File Handler
    fh = logging.FileHandler(fname)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class ContrastiveDataGenerator(Sequence):

    def __init__(self, data_pth='../data', batch_size=10, shuffle=True, seed=42, categories=list(), normalize=True,
                 fs=16000, chunk_size=4096, context_samples=5, contrastive_samples=1):
        """
        Constructor

        :param data_file: path to data file
        :param meta_file:  path to meta file
        :param batch_size: batch size
        :param measurement_ids: list of measurement ids. Dedicated for CV
        :param shuffle:
        :param seed: random seed
        :param test_mode: return samples and signal_ids
        :param normalize: to normalize the data
        """
        self.it = 0
        self.shuffle = shuffle
        self.data_pth = data_pth
        self.normalize = normalize
        self.fs = fs
        self.batch_size = batch_size
        self.seed = seed
        self.context_samples = int(context_samples)
        self.contrastive_samples = int(contrastive_samples)
        self.chunk_size = int(chunk_size)

        # Extract list of files from csv
        file_list = pd.read_csv(os.path.join(data_pth, 'train_curated.csv'))
        if len(categories) == 0:
            self.file_list = file_list
        else:
            self.file_list = file_list.query('labels in @categories').fname.tolist()
        self.list_sz = len(self.file_list)
        self.max_it = int(np.ceil(self.list_sz / self.batch_size))

    def __len__(self):
        return self.max_it

    def on_epoch_end(self):
        """
        Performs at the end of each epoch
        :return:
        """
        l = self.file_list
        shuffle(l)
        self.file_list = l

    def __getitem__(self, item):
        """
        Return one batch
        :param item:
        :return:
        """
        return self.__data_generation(item)

    def __data_generation(self, it):
        """
        Data generator
        :param it:
        :return:
        """
        pos = np.minimum(it * self.batch_size, self.list_sz)
        frames = (self.contrastive_samples+self.context_samples)*self.chunk_size

        i = 0
        context_batch = np.zeros([self.batch_size, self.context_samples, self.chunk_size])
        contrastive_batch = np.zeros([self.batch_size, self.contrastive_samples, self.chunk_size])

        while i < self.batch_size:
            fname = self.file_list[pos]
            pos = (pos+1) % self.list_sz
            signal, sr = librosa.load(os.path.join(self.data_pth, fname), sr=self.fs)
            if signal.shape[0]-frames < 0:
                logging.getLogger(__name__).info(' File {:s} is too short'.format(fname))
            else:
                random_shift = np.random.randint(signal.shape[0]-frames)
                batch = signal[random_shift:(frames + random_shift)].reshape((-1, self.chunk_size), order='C')
                context_batch[i, :, :] = batch[:self.context_samples, :]
                contrastive_batch[i, :, :] = batch[self.context_samples:self.context_samples+self.contrastive_samples, :]
                i +=1

        # generate labels
        labels = None
        return ([context_batch, contrastive_batch], labels)