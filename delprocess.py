# import pandas as pd
import os

os.chdir('../../wav')

i=0
for root, dirs,files in os.walk('.'):
    for name in files:
        line = ''
        if name.endswith('.txt'):
            i += 1
            with open(os.path.join(root,name),'r',encoding='utf-8') as readfile:
                    while True:
                        line = readfile.readline()
                        if not line:
                            break
                        text = line.replace('o', '').replace('/', '').replace('+', '').replace('*',
                                                                                               '').replace('b',
                                                                                                           '').replace(
                            'n', '').replace('l', '')
                        text = text.strip()

                        while (")(" in text):
                            first = text.split(')(')[0]
                            first = first.split('(')[-1]
                            first = '(' + first + ')'
                            text = text.replace(first, '')
                        text = text.replace('(', '').replace(')', '')
            with open(os.path.join(root, name), 'w', encoding='utf-8') as writefile:
                    writefile.write(text)
            if i%1000 == 0 :
                print(name +' is changed!!')

        if name.endswith('.trn'):
            i += 1
            with open(os.path.join(root,name),'r',encoding='utf-8') as readfile:
                with open(os.path.join(root,name[:-4]+'.txt'),'w',encoding='utf-8') as writefile:
                    while True:
                        # for i in range(100):
                        line = readfile.readline()
                        if not line:
                            break
                        front = line.split(' :: ')[0]
                        text = line.split(' :: ')[1]
                        text = text.replace('o', '').replace('/', '').replace('+', '').replace('*',
                                                                                               '').replace('b',
                                                                                                           '').replace(
                            'n', '').replace('l', '')
                        text = text.strip()
                        while (")(" in text):

                            first = text.split(')(')[0]
                            first = first.split('(')[-1]
                            first = '(' + first + ')'
                            text = text.replace(first, '')
                        text = text.replace('(', '').replace(')', '')

                        line = front + " :: " + text
                        writefile.write(line + '\n')
            os.remove(os.path.join(root,name))
            if i%1000 == 0 :
                print(name +' is changed!!')