import os
import json
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Song:
    id: int
    genre: list
    path: str

class MelonGenreDataset:
    def __init__(self, path, k_fold_ver):
        # parse song list
        songs, genre_dict = self.parse_song_list(
            os.path.join(path, 'song_meta.json'), os.path.join(path, 'arena_mel')
        )
        self.genres = list(genre_dict.keys())

        # k_fold
        i = 0
        buffer = [{}, {}, {}]
        w = open(f'{path}/mgc_kfold/ver_{k_fold_ver}.txt', 'r')
        for line in w.readlines():
            for item in line.split(' '):
                try:
                    buffer[i][int(item)] = None
                except:
                    pass
            i += 1
        train_id = buffer[0]
        val_id = buffer[1]
        eval_id = buffer[2]

        # train_set
        self.train_set = self.sample_data(songs, train_id)

        # val_set
        self.val_set = self.sample_data(songs, val_id)

        # eval_set
        self.eval_set = self.sample_data(songs, eval_id)

        # error check
        assert len(self.train_set) == 490234, f'train set: {len(self.train_set)}'
        assert len(self.val_set) == 61270, f'val set: {len(self.val_set)}'
        assert len(self.eval_set) == 61270, f'eval set: {len(self.eval_set)}'
        
    def parse_song_list(self, path_json, path_data):
        """Read JSON file and make song dictionary.
        Note that error data is discarded.
        (Error data: no genre or multi genre)
        """
        songs = []
        genre_dict = {}
        
        # parse json
        with open(path_json, encoding="utf-8") as f:
            for item in tqdm(json.load(f), desc='parse song list', ncols=90):
                # check error data
                if len(item['song_gn_gnr_basket']) == 1:
                    genre = item['song_gn_gnr_basket'][0]
                    
                    # convert genre (str) -> (int)
                    try: genre_dict[genre]
                    except: genre_dict[genre] = len(genre_dict.keys())
                    genre = genre_dict[genre]

                    # append
                    item = Song(
                        id=item['id'],
                        genre=genre,
                        path=os.path.join(path_data, f'{item["id"] // 1000}/{item["id"]}.npy')
                    )
                    songs.append(item)

        return songs, genre_dict
    
    def sample_data(self, songs, ids):
        """Drop songs which not in ids
        """
        datas = []
        for e in songs:
            try:
                ids[e.id]
                datas.append(e)
            except: 
                pass
        return datas