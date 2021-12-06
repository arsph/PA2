import warnings
import pandas as pd
warnings.filterwarnings(action='ignore')
import random

with open('candidate_df.csv', newline='') as csvfile:
    df = pd.read_csv("candidate_df.csv")

# samples = 20
# df = df.head(20)

def Generate_Rand_False(df):

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


output = pd.DataFrame(columns=['term1','term2','related'])

# print(df)
ind = 0
line = 0
for word in df['words']:
    line += 1
    print("In Line: ", line)
    terms = word.split(", ")
    for i in range(len(terms)-1):
        for j in range(i+1, len(terms)):
            output.loc[ind] = [terms[i], terms[j], 1]
            ind+=1
            r1, r2 = Generate_Rand_False(df)
            if (r1 != r2):
                output.loc[ind] = [r1, r2, 0]
                ind+=1


output.to_csv("input_balanced.csv", sep='\t')
# print(output)