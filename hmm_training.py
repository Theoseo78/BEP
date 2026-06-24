# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import check_random_state


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
    # data: numpy array data points
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
                best_model = h
    return best_model

def pull_data(v):
    # v: variable name
    iam_vars = ["st", "ce"]
    nigem_vars = ["pr", "epi", "ltir", "rGDP", "er"]
    var_map = {"st": "AR6 climate diagnostics|Surface Temperature ("
                     "GSAT)|MAGICCv7.5.3|50.0th Percentile",
               "ce": "Emissions",
               "pr": f"Central bank Intervention rate (policy interest rate) ; %",
               "epi": f'Equity prices',
               "ltir": f'Long term interest rate ; %',
               "rGDP": f'Gross Domestic Product (GDP)'}
    if v in iam_vars:
        test_df = pd.read_csv(r"NGFS data/IAM_working_set.csv")
    elif v in nigem_vars:
        test_df = pd.read_csv(r"NGFS data/NiGEM_working_set.csv")
    else:
        raise ValueError("Variable not in list of available variables")
    # Globally looking at stuff
    temp = test_df[test_df["Variable"].str.contains(var_map[v], case=False, regex=False)]
    temp = temp[temp["Region"].str.contains("World")]
    temp = temp.drop(["Model", "Region", "Variable", "Unit", "Unnamed: 0"], axis=1)
    temp = temp.groupby(["Scenario"], as_index=True).mean()
    # Get rid of any N/A values
    if temp.isnull().values.any():
        temp = temp.interpolate(axis=1)
    return temp

def calculate_likelihood(v, train_file, bin_amount, max_comp, test_file=None):
    # v: variable name
    # train_file: csv file to use as training data
    # csv contains only one column
    # bin amount: dictate how many bins the data should have
    # max_comp: max. components present in HMM
    # test_file = csv file to use as test data
    # csv contains 4 rows for 4 scenarios, and observations on the columns

    # Create histogram of data
    #data, bins = create_histogram(train_file, bin_amount)
    # Transform data for HMM training

    # Read csv into dataframe
    df = pd.read_csv(r"Historical data/hmm training/" + train_file)
    data = df.iloc[:, 0].to_numpy()
    data = data.reshape(-1, 1)
    # Find optimal model
    model = find_optimal_model(data, max_comp)

    # Pull relevant data
    if test_file:
        temp = pd.read_csv(r"Historical data/hmm training/" + test_file, index_col="Scenario")
    else:
        # Map to correct NiGEM/IAM variable
        # NiGEM variables depend on scenario
        temp = pull_data(v)


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
        #scenario_data = pd.cut(row, bins=bins, labels=False)
        data = row.to_numpy()
        data = data.reshape(-1, 1)
        z = model.predict(data)
        scores[sc] = model.score(data)

    # Normalize likelihood on sum
    norm_fac = sum(scores.values())
    for k, v in scores.items():
        scores[k] = v / norm_fac
    return scores

def calculate_mix_weights():
    variables = [("st", "mean_surface_temp.csv", 10, 10),
                 ("pem", "historical_pem.csv", 10, 10, "pem_iam.csv"),
                 ("ce", "historical_ce.csv", 10, 10)]
    variable_scores = []
    for vt in variables:
        variable_scores.append(calculate_likelihood(*vt))

    mean_scores = {}
    for d in variable_scores:
        for k, v in d.items():
            if k not in mean_scores.keys():
                mean_scores[k] = v / len(variable_scores)
            else:
                mean_scores[k] += v / len(variable_scores)
    return mean_scores

if __name__ == "__main__":
    data, file = ("", "")
    train, bin = create_histogram("historical_ce.csv", 10)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data)
    ax.set_xlabel(f"{file}")
    ax.set_ylabel("Count")
    plt.show()

    train = train.to_numpy()
    train = train.reshape(-1, 1)
    m = find_optimal_model(train, 10)
    print(m.n_components)

    # Plot state counts of data
    pred = m.predict(train)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(pred, bins=m.n_components)
    ax.set_xlabel("State")
    ax.set_ylabel("Count")
    plt.show()






