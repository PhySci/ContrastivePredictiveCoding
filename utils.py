from keras.utils import Sequence
from random import shuffle
import pandas as pd
import numpy as np
import librosa
import os
import logging
import warnings
from random import shuffle
import numpy as np

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
            self.file_list = file_list.fname.tolist()
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

        # shuffle data
        #idx = np.random.choice(range(self.batch_size), self.batch_size, replace=False)
        #contrastive_batch = contrastive_batch[idx, :, :]
        labels=np.zeros([self.batch_size, self.batch_size])
        labels=np.identity(self.batch_size)
        labels = labels[:, :, np.newaxis]
        #labels[range(self.batch_size), idx] = 1
        s = ([context_batch[:, :, :, np.newaxis], contrastive_batch[:, :, :, np.newaxis]], labels)
        return s