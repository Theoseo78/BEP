# %%
import pandas as pd
import matplotlib.pyplot as plt
import hmmlearn.hmm as hmm
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import check_random_state

# https://wmo.int/resources/dashboards/global-mean-temperature-1850-2024
# Compared to mean in 1850-1900

# %%
df = pd.read_csv(r"Historical data/hmm training/mean_temp.csv")
df.set_index("year", inplace=True)
# Take the mean of the models
df['mean'] = df.mean(axis=1)

# Modify bins to account for values outside of min/max
# Bin the observations in intervals of (a, b])
min_temp, max_temp = df['mean'].min(), df['mean'].max()
bin_amount = 10
in_len = (max_temp - min_temp)/bin_amount
# Define bins
bins = np.linspace(min_temp, max_temp, bin_amount + 1)
bins[0], bins[-1] = -np.inf, np.inf
print(bins)
# Change first and last bins to include -inf, inf respectively
# Use index to refer back to bins so it is compatible with the HMM
df['bins'] = pd.cut(df['mean'], bins=bins, labels=False, right=True)

# TODO implement Hidden markov
# %%
# Show the histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['mean'], bins=10)
ax.set_xlabel("Surface temperature (C)")
ax.set_ylabel("Count")
plt.show()
# %%
data = df["bins"].to_numpy()
data = data.reshape(len(df), 1)

# TODO Optimize amount of states using AIC, BIC and log-likelihood
# Optimizing model for amount of components
rs = check_random_state(546)
max_comp = 10
best_ll, best_model = -np.inf, hmm.GaussianHMM()
for n in range(2, max_comp + 1):
    for i in range(10):
        h = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=100, random_state=rs)
        out = h.fit(data)
        score = h.score(data)
        if best_ll < score:
            best_ll = score
            print(score)
            best_model = h
    print(f"Completed n = {n}")
# %%
# Plot state counts of data
pred = best_model.predict(data)
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(pred, bins=best_model.n_components)
ax.set_xlabel("State")
ax.set_ylabel("Count")
plt.show()
# %%
# Wrangle data to extract necessary data
iam_data = pd.read_csv(r"NGFS data/IAM_working_set.csv")
temp = iam_data[iam_data["Variable"].str.contains("AR6 climate diagnostics|Surface Temperature ("
                                                  "GSAT)|MAGICCv7.5.3|50.0th Percentile", case=False, regex=False)]
temp = temp.drop(["Model", "Region", "Variable", "Unit", "Unnamed: 0"], axis=1)
temp = temp.groupby(["Scenario"], as_index=True).mean()
# %%
# Predict temperature from NGFS and calculate likelihood
scenario_list = [
    "Net Zero 2050",
    "Delayed transition",
    "Current Policies",
    "Fragmented World",
]
scores = {}
for i, row in temp.iterrows():
    sc = str(row.name)
    scenario_data = pd.cut(row, bins=bins, labels=False)
    data = scenario_data.to_numpy()
    data = data.reshape(len(scenario_data), 1)
    states = best_model.predict(data)
    sc_score = best_model.score(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title(sc)
    ax.hist(states, bins=best_model.n_components)
    ax.set_xlabel("State")
    ax.set_ylabel("Count")
    plt.show()

    scores[sc] = best_model.score(data)

# log-likelihood of -11,008
norm_fac = sum(scores.values())
for k, v in scores.items():
    scores[k] = v/norm_fac



