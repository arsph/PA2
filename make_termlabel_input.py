import pandas as pd
import numpy as np
import random

with open('candidate_df/candidate_df.csv', newline='') as csvfile:
    df = pd.read_csv("candidate_df/candidate_df.csv")

data = df
# samples = 5
# df = df.head(samples)

def generate_rand_false_with_label(df):

    row1_id, row2_id, row3_id = random.sample(range(len(df)), 3)

    row1 = df.loc[row1_id]['words'].split(", ")
    row1 = pd.DataFrame(row1)
    rand_word_id_1 = random.sample(range(len(row1)),1)
    rand_word_id_1 = rand_word_id_1[0]
    rand_word_1 = row1.loc[rand_word_id_1][0]

    row2 = df.loc[row2_id]['words'].split(", ")
    row2 = pd.DataFrame(row2)
    rand_word_id_2 = random.sample(range(len(row2)),1)
    rand_word_id_2 = rand_word_id_2[0]
    rand_word_2 = row2.loc[rand_word_id_2][0]

    rand_label = df.loc[row3_id]['labels']

    return(rand_word_1, rand_word_2,rand_label)


positives = pd.DataFrame(columns=['term1','term2','label','related'])
negatives = pd.DataFrame(columns=['term1','term2','label','related'])

row = 0
line = 0
for word in df['words']:
    line += 1
    print("Making positives - line: ", line,"/",len(data))
    terms = word.split(", ")
    for i in range(len(terms)-1):
        if (i > 10):
            break
        for j in range(i+1, len(terms)):
            positives.loc[row] = [terms[i], terms[j], df.loc[line-1]['labels'], 1]
            row+=1

positives = positives.drop_duplicates()

for i in range(len(positives)):
    r1, r2, label = generate_rand_false_with_label(data)
    if( (r1 != r2) ):
        if ((((positives['term1'] == r1) & (positives['term2'] == r2) & (positives['label'] == label)).any())  or
                (((positives['term1'] == r2) & (positives['term2'] == r1) & (positives['label'] == label)).any()) ):
            i-=1
            continue
        negatives.loc[i] = [r1, r2, label, 0]
        if (i%2000 == 0 and i>0):
            print(i, "iterates done.")

negatives = negatives.drop_duplicates()
positives = positives.sample(len(negatives))

print( "Negatives: ", len(negatives), "Positives: ",  len(positives), "are generated")

frames = [positives, negatives]
result = pd.concat(frames)

result.to_csv("input_balanced_with_labels.csv", sep=',', index=False)