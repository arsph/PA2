import warnings
import fasttext
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

ft = fasttext.load_model('D:\cc.de.300.bin')

path = 'Softwaredev/input_term_term.csv'

with open(path, newline='') as csvfile:
    df = pd.read_csv(path)

def concatenate(term1,term2):
    term1 = ft.get_word_vector(term1)
    term2 = ft.get_word_vector(term2)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = term1[i]
    for i in range(300,600):
        word[i] = term2[i-300]
    return word

vectors = []

for i in range(len(df)):
    vectors.append(concatenate(df.loc[i]['term1'],df.loc[i]['term2']))

vectors = np.array(vectors)
# print(type(vectors),vectors.shape)
X_train, X_test, y_train, y_test = train_test_split(vectors, df['related'], test_size=0.2)

model = Sequential([
    keras.layers.Flatten(input_shape=(600,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=8)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# model.save("Softwaredev/model_binary")
model.summary()