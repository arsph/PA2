import numpy as np
import fasttext
import warnings
warnings.filterwarnings('ignore')

model = fasttext.load_model('D:\cc.de.300.bin')

def concatenate(term1,term2):
    term1 = model.get_word_vector(term1)
    term2 = model.get_word_vector(term2)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = term1[i]
    for i in range(300,600):
        word[i] = term2[i-300]