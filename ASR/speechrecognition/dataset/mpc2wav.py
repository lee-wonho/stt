import sys , os , wave
from shutil import copytree

os.chdir('../../')
src_path = 'data'
dst_path = 'wav'

# copy the directory structure
copytree(src_path, dst_path)

os.chdir(dst_path)

for root, dirs, files in os.walk('.'):
    for name in files:
        if name.endswith('.pcm'):
            with open(os.path.join(root,name),'rb') as pcmfile:
                pcmdata = pcmfile.read()
            with wave.open(os.path.join(root,name[:-4]+'.wav'),'wb') as wavfile:
                wavfile.setparams((1,2,16000,0,'NONE','NONE'))
                wavfile.writeframes(pcmdata)
            os.remove(os.path.join(root,name))