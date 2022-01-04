import tensorflow as tf
import numpy as np
import fasttext

ft = fasttext.load_model('D:\cc.de.300.bin')

def concatenate_with_label(term1,term2,l):
    term1 = ft.get_word_vector(term1)
    term2 = ft.get_word_vector(term2)
    label = ft.get_word_vector(l)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = (term1[i]+term2[i])/2
    for i in range(300,600):
        word[i] = label[i-300]
    return word

model = tf.keras.models.load_model('model_binary_with_labels',compile = True)

vector = concatenate_with_label('Arbeitgeber','Unternehmen','Gesellschaft')
vector = vector.reshape(1,600)

result = model.predict(vector)
print(result)
