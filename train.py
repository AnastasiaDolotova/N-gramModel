"""
Модель обучена на концовке 1-ой части романа 'Преступление и наказание' Ф.М. Достоевского
"""

import numpy as np
import re
import pickle
import argparse


def clear_sample(sample: str):
    sample = sample.lower()
    sample = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', sample)
    return sample.split()


def make_dict(ngram: list = None):
    val = 0
    _dict = dict()
    for element in ngram:
        if element not in _dict:
            for i in range(len(ngram)):
                if element == ngram[i]:
                    val += 1
        _dict[element] = val
        val = 0
    return _dict


def make_ngrams(strings: list, n: int):
    ngrams = zip(*[strings[i:] for i in range(n)])
    return list(ngrams)


def load_data(_input: str = None):
    if _input is not None:
        with open(_input, 'r', encoding='utf-8') as train:
            sample = train.read()
    else:
        sample = str(input())
    cleared_sample = clear_sample(sample)
    tokens = list()
    for element in cleared_sample:
        tokens += element.split(' ')
    return list(filter(lambda x: x != '', tokens))


class Model:
    def __init__(self, n=2):
        self.model = dict()
        self.n = n
        self.tokens = None
        self._dict = None

    def fit(self, _input: str = None):
        self.tokens = load_data(_input)
        self._dict = make_dict(self.tokens)
        if self.n == 1:
            self.model = {(uni_gram,): count / len(self.tokens) for uni_gram, count in self._dict.items()}
        else:
            self.model = self.smooth()

    def smooth(self) -> dict:
        size = len(self._dict)
        n_grams = make_ngrams(self.tokens, self.n)
        n_vocab = make_dict(n_grams)
        k_grams = make_ngrams(self.tokens, self.n - 1)
        k_vocab = make_dict(k_grams)

        def count(n_gram, n_count):
            k_gram = n_gram[:-1]
            k_count = k_vocab[k_gram]
            return n_count / (k_count + 0.001 * size)

        return {n_gram: count(n_gram, _count) for n_gram, _count in n_vocab.items()}

    def generate(self, length: int, prefix: str = None) -> str:
        if prefix is None:
            prefix = self.tokens[np.random.randint(0, high=len(self.tokens))]
        res = prefix.split(' ')
        for i in range(length):
            prev = () if self.n == 1 else tuple(res[-(self.n - 1):])
            _next, _ = self.make_best_choice(prev, i)
            res.append(_next)
        return ' '.join(res)

    def make_best_choice(self, prev, i):
        candidates = ((ngram[-1], prob) for ngram, prob in self.model.items() if ngram[:-1] == prev)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return '', 1
        else:
            return candidates[0 if prev != () else i]

    def save_model(self, path: str = './model.pkl'):
        with open(path, 'wb') as _model:
            pickle.dump((self.model, self.tokens), _model)

    def load_model(self, path: str = './model.pkl'):
        with open(path, 'rb') as _model:
            self.model, self.tokens = pickle.load(_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='путь к директории, в которой лежит коллекция документов. Если '
                                                      'данный аргумент не задан, считать, что тексты вводятся из '
                                                      'stdin.')
    parser.add_argument('--model', type=str, help='путь к файлу, в который сохраняется модель.')

    args = parser.parse_args()
    model = Model()

    if args.input_dir is not None:
        model.fit(args.input_dir)
    else:
        model.fit()

    model.save_model(args.model)
