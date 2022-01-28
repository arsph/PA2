import numpy as np
import pandas as pd
import fasttext
import tensorflow as tf

ft = fasttext.load_model('D:\cc.de.300.bin')
model_binary = tf.keras.models.load_model('Softwaredev/model_binary',compile = True)
model_binary_with_labels = tf.keras.models.load_model('Softwaredev/model_binary_with_labels',compile = True)

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

with open('Softwaredev/candidate_df_sortwaredev_with_scores.csv', newline='') as csvfile:
    df = pd.read_csv("Softwaredev/candidate_df_sortwaredev_with_scores.csv")

labels = df['labels']

uncategorized_terms = []
for word in df['words']:
    terms = word.split(", ")
    for i in range(len(terms)-1):
        uncategorized_terms.append(terms[i])
uncategorized_terms = list(dict.fromkeys(uncategorized_terms))

categorized_terms = pd.DataFrame(columns=['labels','words'])

la = ""
row = 0
while(len(uncategorized_terms) > 1):
    term1 = uncategorized_terms.pop(0)
    for i in range(len(uncategorized_terms)):
        term2 = uncategorized_terms[i]
        result = get_term_similarity(term1,term2)
        if((result > 0.5)  and (term1 != term2)):
            # max_similarity = 0.5
            la = ""
            for l in labels:
                sim_label = get_term_with_label_similarity(term1,term2,l)
                if(sim_label > 0.5):
                    # max_similarity = sim_label
                    la = l
                    print(term1,term2,la,sim_label,len(uncategorized_terms),"left.")
                    categorized_terms.loc[row] = [la,term1]
                    categorized_terms.loc[row+1] = [la,term2]
                    row+=2
                    # categorized_terms = categorized_terms.append([la,term2])
                    # add_terms_to_label(term1,term2,la,categorized_data)
                    # uncategorized_terms.remove(term1)
                    uncategorized_terms.remove(term2)
                    break
            if( la == ""):
                continue
            break
        if(i == (len(uncategorized_terms)-1)):
            print("For term:",term1,"cannot find a matched term/label!")
            categorized_terms.loc[row] = ["Uncategorized", term1]
            row+=1

print(len(categorized_terms),categorized_terms)
categorized_terms.to_csv("Softwaredev/categorized_terms_from_scratch.csv", sep=',', index=False)
