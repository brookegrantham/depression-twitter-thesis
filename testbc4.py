import pandas as pd

df = pd.read_csv('reduced_set')
print(df['text'][0])