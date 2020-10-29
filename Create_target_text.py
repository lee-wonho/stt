import pandas as pd

def load_label(filepath):
    char2id = dict()
    id2char = dict()
    ch_labels = pd.read_csv(filepath, encoding='cp949')
    id_list = ch_labels['id']
    char_list = ch_labels['char']
    freq_list = ch_labels['freq']

    for (id, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char]=id
        id2char[id]=char
    return char2id, id2char

def sentence_to_target(sentence, char2id):
    target=""
    for ch in sentence:
        print(ch)
        if ch!='\n':
            target+=(str(char2id[ch])+' ')
    return target[:-1]

def target_to_sentence(target, id2char):
    sentence=""
    targets=target.split()

    for n in targets:
        sentence+=id2char[int(n)]

    return sentence

char2id, id2char = load_label("test_labels.csv")

# fname = ['dev2','eval_clean2','eval_other2','train2']

# for i in range(len(fname)):
sentence, target = None, None
f = open('script_without_title/'+'train2'+'.txt', 'r')
savef = open('label_text.txt','w')
while True:
    line = f.readline()
    if not line:
        break
    target = sentence_to_target(line, char2id)
    savef.write(target)

f.close()
savef.close()
