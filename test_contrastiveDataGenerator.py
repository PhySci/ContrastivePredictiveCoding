from unittest import TestCase
from tqdm import tqdm
from my_cpc.utils import ContrastiveDataGenerator, setup_logging

categories = ['Marimba_and_xylophone', 'Scissors', 'Gong', 'Printer', 'Keys_jangling', 'Zipper_(clothing)',
              'Computer_keyboard', 'Finger_snapping']

class TestContrastiveDataGenerator(TestCase):

    def test_generator(self):
        setup_logging('test_generator.log')
        g = ContrastiveDataGenerator(categories=categories, data_pth='../data/train_curated')

        for i in range(5):
            for e in tqdm(g):
                pass