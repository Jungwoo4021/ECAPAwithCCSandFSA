from dataclasses import dataclass

@dataclass
class Song:
    genre: int  # genre index
    path: str

class GTZAN:
    def __init__(self, path, kfold_ver):
        
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', \
                       'metal', 'pop', 'reggae', 'rock']

        k_fold = []

        for i in range(10):
            n_fold = open(path + '/10_fold/'+str(i+1)+'_fold.txt','r').read().splitlines()
            k_fold.append(n_fold)

        eval_fold = kfold_ver        
        val_fold = (kfold_ver+1) if (kfold_ver+1) <= 10 else 1
        train_fold = [ i+1 for i in range(10)]
        train_fold.remove(eval_fold)
        train_fold.remove(val_fold)

        print(f"train_fold: {train_fold}, val_fold: {val_fold}, eval_fold: {eval_fold}")

        self.eval_list = self.setting_song_data(path,k_fold[eval_fold-1])
        self.val_list = self.setting_song_data(path,k_fold[val_fold-1])
        self.train_list = []
        for k in train_fold:
            self.setting_song_data(path, k_fold[k-1], self.train_list)

        print(f"#train: {len(self.train_list)}, #val: {len(self.val_list)}, #eval: {len(self.eval_list)}")

    def setting_song_data(self, data_path, song_list, songs=None):
        if songs == None:
            songs = []

        for s in song_list:
            song = Song(
                genre=self.genres.index(s.split('/')[1]),
                path=data_path+'/'+s
            )
            songs.append(song)

        return songs

if __name__ == '__main__':
    gtzan = GTZAN('/data/GTZAN', 1)
