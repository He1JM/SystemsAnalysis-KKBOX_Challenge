import pandas as pd
import numpy as np


members = pd.read_csv("data/members.csv/members.csv")
songs = pd.read_csv("data/songs.csv/songs.csv")
extra = pd.read_csv("data/song_extra_info.csv/song_extra_info.csv")
train = pd.read_csv("data/train.csv/train.csv")


members['bd'] = members['bd'].apply(lambda x: np.nan if x <= 0 or x > 100 else x)

members['gender'] = members['gender'].fillna('unknown')

members['registration_init_time'] = pd.to_datetime(
    members['registration_init_time'], format='%Y%m%d', errors='coerce'
)
members['expiration_date'] = pd.to_datetime(
    members['expiration_date'], format='%Y%m%d', errors='coerce'
)


songs['song_length_min'] = songs['song_length'] / 60000

for col in ['genre_ids', 'artist_name', 'composer', 'lyricist']:
    songs[col] = songs[col].fillna('unknown')

songs['num_genres'] = songs['genre_ids'].apply(lambda x: len(str(x).split('|')))

extra['isrc_country'] = extra['isrc'].str[:2]
extra['isrc_year'] = extra['isrc'].str[5:7]

extra['isrc_year'] = extra['isrc_year'].apply(
    lambda x: 2000 + int(x) if str(x).isdigit() else np.nan
)

train.drop_duplicates(subset=['msno', 'song_id'], inplace=True)

df = train.merge(members, on='msno', how='left')
df = df.merge(songs, on='song_id', how='left')
df = df.merge(extra[['song_id', 'isrc_country', 'isrc_year']], on='song_id', how='left')


df = df.dropna(thresh=int(df.shape[1] * 0.5))

df = df.fillna({
    'source_system_tab': 'unknown',
    'source_screen_name': 'unknown',
    'source_type': 'unknown'
})

df.to_csv("clean_kkbox.csv", index=False)
print("Archivo generado")