# %%
import pandas as pd
import matplotlib.pyplot as plt
import hmmlearn.hmm as hmm
from sklearn.model_selection import StratifiedKFold

# https://wmo.int/resources/dashboards/global-mean-temperature-1850-2024
# Compared to mean in 1850-1900

# %%
#TODO Implement cross-validation to optimize amount of components

# %%
df = pd.read_csv(r"Historical data/mean_temp.csv")
df.set_index("year", inplace=True)
df = df.mean(axis=1)



# TODO implement Hidden markov
# %%
# Show the histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df, bins=10)
ax.set_xlabel("Surface temperature (C)")
ax.set_ylabel("Count")
plt.show()

# %%
lst = df.to_numpy()
print(type(lst))
# %%
# Reshaping the data
data = df.to_numpy()
data = data.reshape(len(df), 1)
# Training the HMM model
remodel = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000)
remodel.fit(data)
test = remodel.predict(data)
