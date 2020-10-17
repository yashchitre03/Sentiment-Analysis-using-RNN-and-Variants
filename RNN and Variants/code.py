from pathlib import Path
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

cur_dir = Path.cwd()
dataset_path = cur_dir / 'Dataset' / 'rt-polaritydata'
        
train_X, train_Y, validation_X, validation_Y, test_X, test_Y = [], [], [], [], [], []

def read_and_split(filename, train_X, train_Y, validation_X, validation_Y, test_X, test_Y, label, n=5332):
    with open(str(dataset_path / filename)) as f:
        
        for _ in range(int(0.6*n)):
            train_X.append(f.readline())
            train_Y.append(label)
            
        for _ in range(int(0.6*n), int(0.8*n)):
            validation_X.append(f.readline())
            validation_Y.append(label)
            
        for _ in range(int(0.8*n), n):
            test_X.append(f.readline())
            test_Y.append(label)
            
read_and_split('rt-polarity.pos', train_X, train_Y, validation_X, validation_Y, test_X, test_Y, 1)
read_and_split('rt-polarity.neg', train_X, train_Y, validation_X, validation_Y, test_X, test_Y, 0)

tokenizer = Tokenizer(oov_token='<MISSING>')
tokenizer.fit_on_texts(train_X)

train_seq = tokenizer.texts_to_sequences(train_X)
validation_seq = tokenizer.texts_to_sequences(validation_X)
test_seq = tokenizer.texts_to_sequences(test_X)

maxlen = max(len(x) for x in train_seq)
train_padded = pad_sequences(train_seq, padding='post', maxlen=maxlen)

