import time, math
import pandas as pd


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent) if percent != 0 else 0
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def get_path(filename):
    base_path = '../wav/KsponSpeech/'
    df = pd.read_csv(base_path + filename)
    paths = df['path']

    return paths

def get_train():
    return get_path('train_list.csv')
def get_test():
    return get_path('test_list.csv')


def load_label():
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv('../test_labels.csv', encoding='cp949')
    id_list = ch_labels['id']
    char_list = ch_labels['char']

    for id, char in zip(id_list, char_list):
        char2id[char] = id
        id2char[id] = char

    return char2id, id2char

def sentence_to_target(sentence, char2id):
    target = ""
    for ch in sentence:
        target += (str(char2id[ch]) + ' ')
    return target[:-1]

