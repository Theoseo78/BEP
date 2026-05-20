# %%
import pandas as pd

# https://wmo.int/resources/dashboards/global-mean-temperature-1850-2024
# Compared to mean in 1850-1900

df = pd.read_csv(r"mean_temp.csv")
df.set_index("year", inplace=True)
new_df = df.mean(axis=1)
hist = new_df.hist(bins=10)

# TODO implement Hidden markov

# %%
