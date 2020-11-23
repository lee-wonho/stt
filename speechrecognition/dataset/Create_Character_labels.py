import pandas as pd

PATH  = '../../wav/KsponSpeech_scripts/'

df = pd.DataFrame(columns=['char','freq'])
print('Converting script to DataFrame...')
with open(PATH+'train.txt','r',encoding = 'utf-8')as readfile:
    i = 0
    while True:
        line = readfile.readline()
        if not line:
            break
        i += 1
        front = line.split(' :: ')[0]
        text = line.split(' :: ')[1]
        for ch in text[:-1]:
            if ch not in df['char']:
                df.append({'char':ch,'freq':1},ignore_index= True)
            else:
                df[df['char'] == ch]['freq'] += 1
        if i % 1000 == 0:
            print(i,'lines completed')

df = df.sort_values(by='freq',ascending=False)
df = pd.DataFrame({'char':'</s>','freq':0}).append(df)
df = pd.DataFrame({'char':'<s>','freq':0}).append(df)

print('Success to convert script to DataFrame!!!')
print('Make csv file...')
df.to_csv('../../label/train_labels.csv',encoding = 'utf-8',index = True)