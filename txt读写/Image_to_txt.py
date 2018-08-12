import os
import random

def generate(dir):
    files = os.listdir(dir)
    print('****************')
    print('input :',dir)
    print('start...')
    listText = open('dataset.txt','w')
    random.shuffle(files)  #suhffle list
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = file + '\n'
        listText.write(name)
    listText.close()
    print('down!')
    print('****************')

if __name__ == '__main__':
    generate('test_data/canon/')