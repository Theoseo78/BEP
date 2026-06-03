# %%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv(r"Historical data/hmm training/mean_temp.csv", index_col="year")
new_df = pd.DataFrame(df.mean(axis=1))
new_df.to_csv(r"Historical data/hmm training/mean_surface_temp.csv", index = False)