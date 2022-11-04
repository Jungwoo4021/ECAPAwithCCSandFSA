# FIXME: please set GTZAN_path
"""
README
You don't need to run this file.
Just use "GTZAN/10_fold/*_fold.txt" for k-folding.
If you want to set new k-fold, please run this file.
"""

import glob
import random

def make_k_fold(k=10):
    GTZAN_path = {YOUR_GTZAN_PATH} # ex) '/data/GTZAN'

    dic = {}
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
                       'metal', 'pop', 'reggae', 'rock']

    for genre in genres:
        files = glob.glob(GTZAN_path+'/genres_original/'+genre+'/*.wav')
        random.shuffle(files)
        dic[genre] = files

    N = 10
    for i in range(k):
        f = open(GTZAN_path+'/10_fold/'+str(i+1)+'_fold.txt', 'a')
        for genre in genres:
            files = dic[genre]
            for _ in range(N):
                f.write(files.pop().split('GTZAN/')[-1]+'\n')

if __name__ == '__main__':
    make_k_fold()