import pandas as pd

with open('Softwaredev/categorized_terms_from_scratch.csv', newline='') as csvfile:
    df = pd.read_csv("Softwaredev/categorized_terms_from_scratch.csv")


for i in range(len(df)-1):
    for j in range(i+1,len(df)):
        if(df.loc[i]['labels'] == df.loc[j]['labels']):
            terms = str(df.loc[i]['words']) + ", " + str(df.loc[j]['words'])
            df.loc[i]['words'] = terms
            df.loc[j]['words'] = ""

df = df[df['words'] != ""]

df.to_csv("Softwaredev/categorized_terms_from_scratch.csv", index=False)
