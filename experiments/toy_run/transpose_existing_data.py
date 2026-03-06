import os
import pandas as pd

folder_path = "experiments/toy_run/"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".csv"):
        full_path = os.path.join(folder_path, filename)

        # Read CSV
        df = pd.read_csv(full_path, header=None)

        # Transpose
        df_T = df.transpose()

        # Output path
        out_path = os.path.join(
            folder_path,
            filename.replace(".csv", "_T.csv")
        )

        # Save
        df_T.to_csv(out_path, index=False, header=False)

        print(f"Created: {out_path}")
