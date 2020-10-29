import pandas as pd
from tqdm import trange

# fname = ['dev2','eval_clean2','eval_other2','train2']

label_list = []
label_freq = []
label_df = pd.DataFrame(columns=['char','freq'])
# for i in range(len(fname)):
f = open('script_without_title/'+'train2'+".txt")
while True:
    line = f.readline()
    if not line:
        break
    for ch in line:
        print(ch)
        if ch not in label_list:
            label_list.append(ch)
            label_freq.append(1)
        else:
            label_freq[label_list.index(ch)] += 1
f.close()

label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
for i in range(len(label_list)):
#1번 이하로 등장한 것은 제외하고 train_labels.csv로 만들었고, 1번 이상 등장한 건 test_labels.csv로 만듦.
    if label_freq[i]>1:
        label_df.loc[i] = {'char': label_list[i], 'freq': label_freq[i]}

label_df.to_csv('train_labels.csv', encoding='cp949', index=True)
