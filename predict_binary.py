import tensorflow as tf
import numpy as np
import fasttext

ft = fasttext.load_model('D:\cc.de.300.bin')

def concatenate(term1,term2):
    term1 = ft.get_word_vector(term1)
    term2 = ft.get_word_vector(term2)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = term1[i]
    for i in range(300,600):
        word[i] = term2[i-300]
    return word

model = tf.keras.models.load_model('model_binary',compile = True)

vector = concatenate('Staat','Schule')
vector = vector.reshape(1,600)
# print(vector)
# print(vector.shape)
# Check its architecture
result = model.predict(vector)
print(result.shape)
print(result)
