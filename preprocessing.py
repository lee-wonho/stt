# import pandas as pd

f = open("eval_other.txt", 'r', encoding='utf-8')
f1 = open('eval_other2.txt', 'w')
i = 0
while True:
    # for i in range(100):
    line = f.readline()
    if not line:
        break
    front = line.split(' :: ')[0]
    text = line.split(' :: ')[1]
    text = text.replace('o', '').replace('/', '').replace('+', '').replace('*',
                                                                           '').replace('b', '').replace('n', '').replace('l', '')
    # text = text.replace(' ', '')
    text = text.strip()
    # text = '잠만 기달려봐, 지금. (11시)(열 한 시). 어차피 니 (11시)(열 한 시) (20분)(이십 분)까지 도착해야 되는 거지?'
    while (")(" in text):
        # print(text + ", ")
        first = text.split(')(')[0]
        first = first.split('(')[-1]
        first = '(' + first + ')'
        text = text.replace(first, '')
    text = text.replace('(', '').replace(')', '')
    # print(text)
    line = front + " :: " + text
    f1.write(line + '\n')
    if i % 20 == 0:
        print(line)
    i += 1

f.close()
f1.close()
