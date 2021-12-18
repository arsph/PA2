import pandas as pd
import numpy as np
import random

with open('candidate_df/candidate_df.csv', newline='') as csvfile:
    df = pd.read_csv("candidate_df/candidate_df.csv")

data = df
# samples = 5
# df = df.head(samples)

def generate_rand_false(df):

    row1_id, row2_id = random.sample(range(len(df)), 2)

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

    return(rand_word_1, rand_word_2)

positives = pd.DataFrame(columns=['term1','term2','related'])
negatives = pd.DataFrame(columns=['term1','term2','related'])

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
            positives.loc[row] = [terms[i], terms[j], 1]
            row+=1

positives = positives.drop_duplicates()

for i in range(len(positives)):
    r1, r2 = generate_rand_false(data)
    if( (r1 != r2) ):
        if ((((positives['term1'] == r1) & (positives['term2'] == r2)).any())  or
                (((positives['term1'] == r2) & (positives['term2'] == r1)).any()) ):
            i-=1
            continue
        negatives.loc[i] = [r1, r2, 0]
        if (i%2001 == 1):
            print(i, "iterates done.")

negatives = negatives.drop_duplicates()
positives = positives.sample(len(negatives))

print( "Negatives: ", len(negatives), "Positives: ",  len(positives), "are generated")

frames = [positives, negatives]
result = pd.concat(frames)

result.to_csv("input_balanced.csv", sep=',', index=False)