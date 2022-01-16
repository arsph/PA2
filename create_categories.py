import numpy as np
import pandas as pd
import fasttext
import tensorflow as tf

ft = fasttext.load_model('D:\cc.de.300.bin')
model_binary = tf.keras.models.load_model('model_binary',compile = True)
model_binary_with_labels = tf.keras.models.load_model('model_binary_with_labels',compile = True)

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

def get_term_with_label_similarity(term1,term2,l):
    term1 = ft.get_word_vector(term1)
    term2 = ft.get_word_vector(term2)
    label = ft.get_word_vector(l)

    word = np.zeros(shape=(600,))
    for i in range(300):
        word[i] = (term1[i]+term2[i])/2
    for i in range(300,600):
        word[i] = label[i-300]

    vector = word.reshape(1, 600)
    result = model_binary.predict(vector)
    result = result[0, 0]
    return result

with open('candidate_df/candidate_df_with_scores.csv', newline='') as csvfile:
    df = pd.read_csv("candidate_df/candidate_df_with_scores.csv")

data = df.tail(10)

categorized_terms = pd.DataFrame(columns=['labels','terms'])
line = 0
uncategorized_terms = []
for word in data['words']:
    line += 1
    print("Making uncategorized terms ", line,"/",len(data))
    terms = word.split(", ")
    for i in range(len(terms)-1):
        uncategorized_terms.append(terms[i])

labels = df['labels']

best_label = ""
for term1 in uncategorized_terms:
    max_similarity = 0
    for i in range(len(uncategorized_terms)-1):
        term2 = uncategorized_terms[i+1]
        result = get_term_similarity(term1,term2)
        if(result >= 0.5):
            for label in labels:
                result_with_labels = get_term_with_label_similarity(term1,term2,label)
                if((result_with_labels > max_similarity) and (term1 != term2)):
                    max_similarity = result_with_labels
                    best_label = label
    print(term1,term2,best_label,max_similarity)
                #     print(term1,term2, label, result_with_labels)
                    # if((categorized_terms['labels'] == label).any()):
                    #     new_terms = ", " + term1 + ", " + term2
                    #     old_terms = categorized_terms.loc[categorized_terms['labels'] == label]['terms'].values[0]
                    #     categorized_terms.loc[categorized_terms['labels'] == label]['terms'] = [old_terms , new_terms]
                    # else:
                    #     categorized_terms.append([label, term1 + ", " + term2])


print(categorized_terms)
