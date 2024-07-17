import pandas as pd


df=pd.read_csv('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Generated/combined_output_full_segregated.csv')
filtered_df = df[df['answer2'].str.contains('Category: Can be answered')]
filtered_df.to_csv('/Users/samveg.shah/Desktop/Coverage_Questions/Coverage_Question_Answering/Dataset/full_data_can_2.csv', index=False)
print(len(filtered_df))