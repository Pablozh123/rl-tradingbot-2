import pandas as pd

df = pd.read_csv('data/preprocessed_data.csv')

print("Total Rows:", len(df))
print("SFP Longs:", df['is_sfp_long'].sum())
print("SFP Shorts:", df['is_sfp_short'].sum())
print("GP Longs (<0.01):", (df['dist_to_gp_long'] < 0.01).sum())
print("GP Shorts (<0.01):", (df['dist_to_gp_short'] < 0.01).sum())
