import pandas as pd
import random
import os
from os.path import getsize

def foldernum_padding(filenum):
    if filenum<10:
        return '000'+str(filenum)
    elif filenum<100:
        return '00'+str(filenum)
    elif filenum<1000:
        return '0'+str(filenum)
    else:
        return str(filenum)

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

TRAIN_NUM=int(622545*0.98) #98퍼센트를 학습데이터로
TEST_NUM = 622545-TRAIN_NUM

cfnum=0
fnum=0

BASE_PATH = "KsponSpeech_"

audio_paths=[]
target_paths=[]

for i in range(1,6):
    os.chdir(BASE_PATH+"0"+str(i))
    if i !=5:
        for j in range(0,125):
            cfnum+=1
            os.chdir(BASE_PATH+foldernum_padding(cfnum))
            for k in range(0,1000):
                fnum+=1
                txt = BASE_PATH + filenum_padding(fnum) + '.txt'
                f = open(txt, 'r')
                s = f.readline()
                if len(s)<=100:
                    audio_paths.append(BASE_PATH + filenum_padding(fnum) + ".wav")
                    target_paths.append(BASE_PATH + filenum_padding(fnum) + '.txt')
                f.close()
            os.chdir("../")
        os.chdir("../")
    else:
        for j in range(0,127):
            cfnum+=1
            os.chdir(BASE_PATH+foldernum_padding(cfnum))
            for k in range(0,1000):
                fnum+=1
                txt = BASE_PATH + filenum_padding(fnum) + '.txt'
                f = open(txt, 'r')
                s = f.readline()
                if len(s) <= 100:
                    audio_paths.append(BASE_PATH + filenum_padding(fnum) + ".wav")
                    target_paths.append(BASE_PATH + filenum_padding(fnum) + '.txt')
                f.close()
            os.chdir("../")
        os.chdir("../")

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

for idx in range(len(audio_paths)):
    audio=audio_paths[idx]
    target=target_paths[idx]

    if len(train_dict['audio'])==TRAIN_NUM:
        train_full = True

    if train_full:
        test_dict['audio'].append(audio)
        test_dict['label'].append(target)

    else:
        train_dict['audio'].append(audio)
        train_dict['label'].append(target)


test_df = pd.DataFrame(test_dict)
train_df = pd.DataFrame(train_dict)

test_df.to_csv("test_list.csv", encoding='utf-8', index=False)
train_df.to_csv("train_list.csv", encoding='utf-8', index=False)