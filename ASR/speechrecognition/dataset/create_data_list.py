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

TOTAL_NUM = 622545

cfnum=0
fnum=0

os.chdir('../../wav/KsponSpeech')
BASE_PATH = "KsponSpeech_"

data_paths=[]
target_paths=[]

for i in range(1,6):
    root = BASE_PATH+"0"+str(i)
    if i !=5:
        for j in range(124):
            cfnum+=1
            directory = BASE_PATH+foldernum_padding(cfnum)
            for k in range(0,1000):
                fnum+=1
                txt = BASE_PATH + filenum_padding(fnum) + '.txt'
                path = os.path.join(root+'/'+directory+'/'+ txt)
                f = open(path, 'r', encoding='utf-8')
                s = f.readline()
                if len(s)<=100:
                    data_paths.append(path[:-4])
                else:
                    TOTAL_NUM-=1
                f.close()

    else:
        for j in range(127):
            cfnum+=1
            directory = BASE_PATH+foldernum_padding(cfnum)
            if j == 126:
                for k in range(545):
                    fnum += 1
                    txt = BASE_PATH + filenum_padding(fnum) + '.txt'
                    path = os.path.join(root + '/' + directory + '/' + txt)
                    f = open(path, 'r', encoding='utf-8')
                    s = f.readline()
                    if len(s) <= 100:
                        data_paths.append(path[:-4])
                    else:
                        TOTAL_NUM -=1
                    f.close()
            else:
                for k in range(0, 1000):
                    fnum += 1
                    txt = BASE_PATH + filenum_padding(fnum) + '.txt'
                    path = os.path.join(root + '/' + directory + '/' + txt)
                    f = open(path, 'r', encoding='utf-8')
                    s = f.readline()
                    if len(s) <= 100:
                        data_paths.append(path[:-4])
                    else:
                        TOTAL_NUM -=1
                    f.close()


TRAIN_NUM = int(TOTAL_NUM*0.98) #98퍼센트를 학습데이터로
TEST_NUM = TOTAL_NUM-TRAIN_NUM



random.shuffle(data_paths)

train_full=False
train_dict={
    'path':[]
}
test_dict={
    'path':[]
}

for idx in range(len(data_paths)):
    target= data_paths[idx]


    if len(train_dict['path'])==TRAIN_NUM:
        train_full = True

    if train_full:
        test_dict['path'].append(target)
    else:
        train_dict['path'].append(target)


test_df = pd.DataFrame(test_dict)
train_df = pd.DataFrame(train_dict)

test_df.to_csv("test_list.csv", encoding='utf-8', index=False)
train_df.to_csv("train_list.csv", encoding='utf-8', index=False)