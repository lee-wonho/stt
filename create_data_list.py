import pandas as pd
from tqdm import trange
import random

def filenum_padding(filenum):
    if filenum < 10:
        return '00000' + str(filenum)
    elif filenum < 100:
        return '0000' + str(filenum)
    elif filenum < 1000:
        return '000' + str(filenum)
    elif filenum < 10000:
        return '00' + str(filenum)
    elif filenum < 100000:
        return '0' + str(filenum)
    else:
        return str(filenum)

TOTAL_NUM=622545
TRAIN_NUM=int(622545*0.98) #98퍼센트를 학습데이터로
TEST_NUM = TOTAL_NUM-TRAIN_NUM

aihub_labels = pd.read_csv("test_labels.csv", encoding='cp949')
rare_labels=aihub_labels['char'][2036:]

train_data_list = {'audio':[], 'label':[]}
test_data_list = {'audio':[],'label':[]}

aihub_labels = pd.read_csv('test_labels.csv', encoding='cp949')
rare_labels = aihub_labels['char'][2037:]

fname='KsponSpeech_'
target_fname = 'KsponScript_'

audio_paths=[]
target_paths=[]

for filenum in trange(1, TOTAL_NUM):
    audio_paths.append(fname + filenum_padding(filenum)+".wav")
    target_paths.append(target_fname + filenum_padding(filenum)+'.txt')

data_paths = list(zip(audio_paths, target_paths))
random.shuffle(data_paths)
audio_paths, target_paths = zip(*data_paths)

train_full=False
train_dict={
    'audio':[],
    'label':[]
}
test_dict={
    'audio':[],
    'label':[]
}

PATH = 'C:\\Users\\khak1\\Desktop\\2020년\\광인사\\[라젠]STT 기업 프로젝트\\data\\wav\\KsponSpeech\\'

for idx in trange(len(audio_paths)):
    audio=audio_paths[idx]
    target=target_paths[idx]

    if len(train_dict['audio'])==TRAIN_NUM:
        train_full = True

    if train_full:
        test_dict['audio'].append(audio)
        test_dict['label'].append(target)

    else:
        rare_in = False
        sentence=None
        with open(PATH+(audio).split('.')[0]+".txt")as f:
            sentence = f.readline()

        for rare in rare_labels:
            if rare in sentence:
                rare_in=True
                break
        if rare_in:
            test_dict['audio'].append(audio)
            test_dict['label'].append(target)

        else:
            train_dict['audio'].append(audio)
            train_dict['label'].append(target)


test_df = pd.DataFrame(test_dict)
train_df = pd.DataFrame(train_dict)

test_df.to_csv("test_list.csv", encoding='cp949', index=False)
train_df.to_csv("train_list.csv", encoding='cp949', index=False)