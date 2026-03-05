import pandas as pd

# read existing CSV
df = pd.read_csv(r"c:\UT\382V\MLP_Data.csv")

# transpose dataframe (set first column as index to keep ID label)
df_t = df.set_index('ID').T

# write out to new file
out_path = r"c:\UT\382V\MLP_Data_transposed.csv"
df_t.to_csv(out_path, index=True)

print(f"Transposed data written to {out_path}")
