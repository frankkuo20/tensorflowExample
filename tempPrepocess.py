import pandas as pd

train = pd.read_csv('csv/train.csv')
trainLen = len(train)
print(trainLen)
TRAIN_RATE, TEST_RATE = 0.8, 0.2
TRAIN_NUM = int(trainLen * TRAIN_RATE)

coulmns = [
    'msno',
    'song_id',
    'source_system_tab',
    'source_screen_name',
    'source_type'
]

# Fill in missing Cabin with None
for coulmn in coulmns:
    train[coulmn] = train[coulmn].fillna("None")

train2 = train.iloc[0: TRAIN_NUM]
print(len(train2))
train2.to_csv("./csv/train_train.csv", index=False)

train3 = train.iloc[TRAIN_NUM:]
print(len(train3))
train3.to_csv("./csv/train_test.csv", index=False)

# train2 = train.iloc[0: 500000]
# print(len(train2))
# train2.to_csv("./csv/test22_1.csv", index=False)
#
# train2 = train.iloc[500000: 1000000]
# print(len(train2))
# train2.to_csv("./csv/test22_2.csv", index=False)
#
# train2 = train.iloc[1000000:1500000]
# print(len(train2))
# train2.to_csv("./csv/test22_3.csv", index=False)
#
# train2 = train.iloc[1500000:2000000]
# print(len(train2))
# train2.to_csv("./csv/test22_4.csv", index=False)
#
# train2 = train.iloc[2000000:]
# print(len(train2))
# train2.to_csv("./csv/test22_5.csv", index=False)

# TRAIN_NUM = 500000
# train2 = train.iloc[0: TRAIN_NUM]
# test = train.iloc[TRAIN_NUM:600000]
# print(len(train2))
# print(len(test))
# train2.to_csv("./csv/train2.csv", index=False)
# test.to_csv("./csv/test2.csv", index=False)


# train2 = train[['source_system_tab', 'source_screen_name', 'source_type', 'target']]
# # df1 = pd.get_dummies(train2['source_system_tab'])
# # print(df1.sample(5))
# TRAIN_NUM = 500000
# train3 = train2.iloc[0: TRAIN_NUM]
# train3.to_csv("./csv/dnn/train.csv", index=False)
#
# train4 = train2.iloc[TRAIN_NUM:600000]
# print(len(train4))
# train4.to_csv("./csv/dnn/test.csv", index=False)

# # rnn
# train2 = train[['source_system_tab', 'source_screen_name', 'source_type', 'target']]
# TRAIN_NUM = 500000
# train3 = train2.iloc[0: TRAIN_NUM]
# train3.to_csv("./csv/rnn/train.csv", index=False)
# train4 = train2.iloc[TRAIN_NUM:600000]
# print(len(train4))
# train4.to_csv("./csv/rnn/test.csv", index=False)
