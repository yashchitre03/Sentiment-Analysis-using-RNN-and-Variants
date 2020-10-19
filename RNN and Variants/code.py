from pathlib import Path
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

cur_dir = Path.cwd()
dataset_path = cur_dir / 'Dataset' / 'rt-polaritydata'
embeddings_path = cur_dir / 'word_representations' / 'glove.6B' / 'glove.6B.300d.txt'
        
train_X, train_Y, validation_X, validation_Y, test_X, test_Y = [], [], [], [], [], []

def read_and_split(filename, 
                   train_X, train_Y, 
                   validation_X, validation_Y, 
                   test_X, test_Y, label, n=5332):
    with open(str(dataset_path / filename), 'rt') as f:
        
        for _ in range(int(0.6*n)):
            train_X.append(f.readline())
            train_Y.append(label)
            
        for _ in range(int(0.6*n), int(0.8*n)):
            validation_X.append(f.readline())
            validation_Y.append(label)
            
        for _ in range(int(0.8*n), n):
            test_X.append(f.readline())
            test_Y.append(label)
            
read_and_split('rt-polarity.pos',
               train_X, train_Y,
               validation_X, validation_Y,
               test_X, test_Y, label=1)
read_and_split('rt-polarity.neg', 
               train_X, train_Y, 
               validation_X, validation_Y, 
               test_X, test_Y, label=0)

train_Y = np.asarray(train_Y, dtype=np.int8).reshape((-1, 1))
validation_Y = np.asarray(validation_Y, dtype=np.int8).reshape((-1, 1))
test_Y = np.asarray(test_Y, dtype=np.int8).reshape((-1, 1))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

train_seq = tokenizer.texts_to_sequences(train_X)
validation_seq = tokenizer.texts_to_sequences(validation_X)
test_seq = tokenizer.texts_to_sequences(test_X)

MAX_LEN = max(len(x) for x in train_seq)
train_padded = pad_sequences(train_seq, padding='post', maxlen=MAX_LEN)
validation_padded = pad_sequences(validation_seq, padding='post', maxlen=MAX_LEN)
test_padded = pad_sequences(test_seq, padding='post', maxlen=MAX_LEN)

embeddings_index = {}
with open(str(embeddings_path), 'rt', encoding='utf-8') as f:
    for line in f.readlines():
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs       
        
embedding_matrix = np.zeros((len(tokenizer.word_index)+1, 300))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index, :] = embedding_vector
        
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, 
                    output_dim=300, 
                    embeddings_initializer=Constant(embedding_matrix),
                    mask_zero=True,
                    input_length=MAX_LEN,
                    trainable=False))

option = int(input('Which RNN variant to run? '))
if option == 1:
    model.add(SimpleRNN(units=200))
elif option == 2:
    model.add(LSTM(units=128))
else:
    model.add(GRU(units=128))
    
model.add(Dense(units=1,
                activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
print(model.summary())

stopping_criteria = EarlyStopping(patience=2,
                   verbose=1)

model.fit(x=train_padded, 
          y=train_Y,
          batch_size=256,
          epochs=15,
          callbacks=(stopping_criteria),
          validation_data=(validation_padded, validation_Y))

print('\nResult on the test set:')
model.evaluate(x=test_padded, y=test_Y)

test_pred = (model.predict(test_padded) > 0.5).astype(dtype=np.int8)

precision, recall, fscore, _ = precision_recall_fscore_support(y_true=test_Y,
                                                               y_pred=test_pred,
                                                               average='binary')