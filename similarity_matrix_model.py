import tensorflow as tf
import numpy as np
import pandas as pd
import fasttext

def concatenate(term1,term2):
    term1 = ft.get_word_vector(term1)
    term2 = ft.get_word_vector(term2)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = term1[i]
    for i in range(300,600):
        word[i] = term2[i-300]
    return word

ft = fasttext.load_model('D:\cc.de.300.bin')
model = tf.keras.models.load_model('model_binary_2',compile = True)

with open('candidate_df/candidate_df.csv', newline='') as csvfile:
    df = pd.read_csv("candidate_df/candidate_df.csv")

# df = df.tail(5)
data = df
similarity_scores = []
line = 0
for word in df['words']:
    line += 1
    score = 0
    counter = 0
    print("Making sim. scores - line: ", line,"/",len(data))
    terms = word.split(", ")
    for i in range(len(terms)-1):
        for j in range(i+1, len(terms)):

            vector = concatenate(terms[i], terms[j])
            vector = vector.reshape(1, 600)
            result = model.predict(vector)
            score+= result[0,0]
            counter+=1

    similarity_scores.append(score/counter)

df['scores'] = similarity_scores
df = df.sort_values('scores', ascending=False)

df.to_csv("candidate_df/candidate_df_with_scores.csv", sep=',', index=False)