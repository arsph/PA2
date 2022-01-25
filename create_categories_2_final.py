import pandas as pd
import numpy as np
import fasttext
import tensorflow as tf

with open('Processing/categorized_terms.csv', newline='') as csvfile:
    testdf = pd.read_csv("Processing/categorized_terms.csv")

with open('Processing/candidate_df_processing_with_scores.csv', newline='') as csvfile:
    df = pd.read_csv("Processing/candidate_df_processing_with_scores.csv")

ft = fasttext.load_model('D:\cc.de.300.bin')
model_binary = tf.keras.models.load_model('Processing/model_binary',compile = True)

def get_term_similarity(term1,term2):
    term1 = ft.get_word_vector(term1)
    term2 = ft.get_word_vector(term2)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = term1[i]
    for i in range(300,600):
        word[i] = term2[i-300]

    vector = word.reshape(1, 600)
    result = model_binary.predict(vector)
    result = result[0, 0]
    return result

def find_label_for_term(term,data):
    best_match_label = ""
    max_sim = 0

    for i in range(len(data)):
        label = data.loc[i]['labels']
        words = data.iloc[i]['words']
        words = words.split(", ")
        for j in range(len(words)):
            if (words[j] != term):
                sim = get_term_similarity(term, words[j])
                if (sim > max_sim):
                    max_sim = sim
                    best_match_label = label

    return best_match_label

treshold = 0.6
df = df[df['scores'] >= treshold]

for i in range(len(testdf)):
    term = testdf.loc[i]['words']
    label = testdf.loc[i]['labels']
    print(term, "and", label, ":")
    if( label == '-'):
        label = find_label_for_term(term,df)

    old_words = df[df['labels'] == label]['words']
    ind = old_words.index
    df.at[ind, 'words'] = df[df['labels'] == label]['words'] + ", " + term

df.to_csv("Processing/categorized_terms_final.csv", sep=',', index=False)
