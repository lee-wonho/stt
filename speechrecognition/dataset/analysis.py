import os
import matplotlib.pyplot as plt
from feature import load_audio, get_feature
import numpy as np


i=0

target_lengths = []

for root, dirs, files in os.walk('../../wav/KsponSpeech'):
    for name in files:
       if name.endswith('.wav'):
           i+=1
           wave = load_audio(os.path.join(root,name))
           data = get_feature(wave)
           length = data.shape[0]
           target_lengths.append(length)
           if i % 10000 == 0:
               print(i, 'completed!!!')


plt.boxplot(target_lengths)
plt.title('Text')
plt.show()


lengths = np.asarray(target_lengths)

print(np.quantile(lengths, 0.5),np.quantile(lengths, 0.95))