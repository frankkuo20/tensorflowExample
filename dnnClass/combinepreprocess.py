import pandas as pd

train_train = pd.read_csv('../csv/train_test.csv')

members = pd.read_csv('../csv/members.csv')

songs = pd.read_csv('../csv/songs.csv')

trainLen = len(train_train)
print(trainLen)

coulmns = [
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type'
]
# city,bd,gender,registered_via,registration_init_time,expiration_date
mergeDF = pd.merge(train_train, members, on='msno', how='inner')

mergeDF = pd.merge(mergeDF, songs, on='song_id', how='inner')

print(mergeDF.head())
mergeDF.to_csv("../csv/train_test2.csv", index=False)
