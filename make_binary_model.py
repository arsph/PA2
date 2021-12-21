import warnings
import fasttext
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

model = fasttext.load_model('D:\cc.de.300.bin')

with open('input_balanced.csv', newline='') as csvfile:
    df = pd.read_csv("input_balanced.csv")

def concatenate(term1,term2):
    term1 = model.get_word_vector(term1)
    term2 = model.get_word_vector(term2)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = term1[i]
    for i in range(300,600):
        word[i] = term2[i-300]
    return word

vectors = np.arange(len(df)*600).reshape(len(df),600)

for i in range(len(df)):
    vectors[i] = concatenate(df.loc[i]['term1'],df.loc[i]['term2'])


X_train, X_test, y_train, y_test = train_test_split(vectors, df['related'], test_size=0.3)
# x_training_data = vectors
# y_training_data = df['related']

rnn = Sequential()
rnn.add(LSTM(128, return_sequences = True, input_shape = (600,1)))
rnn.add(Dropout(0.2))
rnn.add(Dense(units = 1, activation='sigmoid'))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
rnn.fit(X_train, y_train, epochs = 2, batch_size = 128, validation_data=(X_test, y_test))
rnn.save("model_binary")
rnn.summary()