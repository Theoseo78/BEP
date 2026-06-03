# %%
import pandas as pd
import matplotlib.pyplot as plt
import hmmlearn.hmm as hmm
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import check_random_state

# Surface temperature
# https://wmo.int/resources/dashboards/global-mean-temperature-1850-2024
# Compared to mean in 1850-1900

# %%
def create_histogram(file, bin_amount):
    # file: csv file to use as training data
    # csv contains only one column
    # bin amount: dictate how many bins the data should have

    # Read csv into dataframe
    df = pd.read_csv(r"Historical data/hmm training/" + file)
    # Construct bins for histogram
    data = df.iloc[:, 0]
    min_val, max_val = data.min(), data.max()
    bins = np.linspace(min_val, max_val, bin_amount + 1)
    bins[0], bins[-1] = -np.inf, np.inf
    # Create column with bins based on row value

    df['bins'] = pd.cut(data, bins=bins, labels=False, right=True)
    return df['bins'], bins


def find_optimal_model(data, max_comp):
    # data: numpy array of histogram bins
    # max_comp: max amount components to use for HMM

    # Set random seed for reproduction
    rs = check_random_state(546)
    best_ll, best_model = -np.inf, GaussianHMM()
    for n in range(2, max_comp + 1):
        # Sample multiple times per same value of n
        for i in range(10):
            h = GaussianHMM(n_components=n, covariance_type="full", n_iter=100, random_state=rs)
            try:
                h.fit(data)
                score = h.score(data)
            except:
                score = -np.inf
            # Look for highest log-likelihood
            if best_ll < score:
                best_ll = score
                print(score)
                best_model = h
        print(f"Completed n = {n}")
    return best_model


def calculate_likelihood(v, file, bin_amount, max_comp):
    # v: variable name
    # file: csv file to use as training data
    # csv contains only one column
    # bin amount: dictate how many bins the data should have
    # max_comp: max. components present in HMM

    # Create histogram of data
    data, bins = create_histogram(file, bin_amount)
    # Transform data for HMM training
    data = data.to_numpy()
    data = data.reshape(len(data), 1)
    # Find optimal model
    model = find_optimal_model(data, max_comp)
    # Map to correct NiGEM/IAM variable
    # NiGEM variables depend on scenario

    iam_vars = ["st", ]
    nigem_vars = ["pr", "epi", "ltir", "rGDP", "er"]
    var_map = {"st": "AR6 climate diagnostics|Surface Temperature ("
                     "GSAT)|MAGICCv7.5.3|50.0th Percentile",
               "pr": f"Central bank Intervention rate (policy interest rate) ; %",
               "epi": f'Equity prices',
               "ltir": f'Long term interest rate ; %',
               "rGDP": f'Gross Domestic Product (GDP)'}

    # Pull relevant data
    if v in iam_vars:
        test_df = pd.read_csv(r"NGFS data/IAM_working_set.csv")
    elif v in nigem_vars:
        test_df = pd.read_csv(r"NGFS data/NiGEM_working_set.csv")
    else:
        raise ValueError("Variable not in list of available variables")
    temp = test_df[test_df["Variable"].str.contains(var_map[v], case=False, regex=False)]
    temp = temp.drop(["Model", "Region", "Variable", "Unit", "Unnamed: 0"], axis=1)
    temp = temp.groupby(["Scenario"], as_index=True).mean()

    # Calculate log-likelihood per scenario
    scenario_list = [
        "Net Zero 2050",
        "Delayed transition",
        "Current Policies",
        "Fragmented World",
    ]
    scores = {}
    for i, row in temp.iterrows():
        sc = str(row.name)
        if sc not in scenario_list:
            raise ValueError("Unexpected row name")
        scenario_data = pd.cut(row, bins=bins, labels=False)
        data = scenario_data.to_numpy()
        data = data.reshape(len(scenario_data), 1)
        z = model.predict(data)
        scores[sc] = model.score(data)

    # Normalize likelihood on sum
    norm_fac = sum(scores.values())
    for k, v in scores.items():
        scores[k] = v / norm_fac
    return scores

# %%
train, bin = create_histogram("mean_surface_temp.csv", 10)
train = train.to_numpy()
train = train.reshape(-1, 1)
m = find_optimal_model(train, 10)
print(m.n_components)
# %%
# Plot state counts of data
pred = m.predict(train)
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(pred, bins=m.n_components)
ax.set_xlabel("State")
ax.set_ylabel("Count")
plt.show()

# %%
scores = calculate_likelihood("st", "mean_surface_temp.csv", 10, 10)



