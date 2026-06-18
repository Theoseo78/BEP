# Main file for working with all data
import pandas as pd
import numpy as np
import time
from models import create_model, ngfs_pull
from hmm_training import find_optimal_model, calculate_likelihood
from sklearn.utils import check_random_state
from scipy.spatial.distance import mahalanobis

# %%
# Initialize variables
rs = check_random_state(547362)
region_lst = ['US']
scenario_list = [
    "Net Zero 2050",
    "Delayed transition",
    "Current Policies",
    "Fragmented World",
]
# This also determines the order of the assets in the numpy arrays
stocks = {'US': [('SP500', 'index'),
                 ('DGS10', 'govbond'),
                 ('BAMLC0A4CBBBEY', 'index')], }

# Train regression models
# One for predictions, one for sensitivity analysis
models = {}
for r in stocks.keys():
    models[r] = {}
    for prod in stocks[r]:
        models[r][prod[0]] = \
            (create_model(prod[0], r, False, ac=prod[1]), create_model(prod[0], r, True, ac=prod[1]))

# Initialize variables for creating the posterior
J = 20000
t_end = 2050

# %%
# Finds optimal HMM per region
opt_hmm = {}
for r in region_lst:
    df = pd.read_csv(f"Historical data/{r}_economical_historical.csv")
    df = df.drop(df.columns[0], axis=1)

    # Reshape to (observations, variables)
    data_arr = df.to_numpy().reshape(-1, len(df.columns))
    # Construct HMM model on the data
    opt_hmm[r] = find_optimal_model(data_arr, 10)


# %%
def roll_macro(region, npaths, horizon):
    # region: what geographical region to consider
    # npaths: Number of paths to be simulated
    # horizon: up to and including what year should be simulated

    # Creates historical asset paths based on historical data rolled forward from an HMM trained on it

    # Calculate length of each path based on specified horizon
    n_samp = (horizon - 2026 + 1) * 12

    # Create numpy array with size (number of years, samples per path, number of paths)
    # Data aggregated on yearly average
    # Keeps track of rolled forward macroeconomic variables
    p_arr = np.zeros(shape=(horizon - 2026 + 1, len(df.columns), npaths))

    # Create numpy array with keeping track of means drawn from model.predict
    # Shape = (assets, years, paths)
    m_arr = np.zeros(shape=(len(models[region]), horizon - 2026 + 1, npaths))
    start_time = time.time()

    # Generate paths based on historical data
    for i in range(npaths):
        path, states = opt_hmm[region].sample(n_samples=n_samp, random_state=rs)
        # Aggregate the path to yearly data
        yearly_path = path.reshape((horizon - 2026 + 1), 12, path.shape[1]).mean(axis=1)
        p_arr[:, :, i] = yearly_path
        df_path = pd.DataFrame(yearly_path, columns=df.columns)
        v = 0
        for mname, m in models[region].items():
            # Predict outcomes and standard deviations of those outcomes
            m_arr[v, :, i] = np.exp(m[0].predict(df_path))
            v += 1
    # Save the arrays
    np.save(f"Prior distributions/{region}_hist_forward", p_arr)
    np.save(f"Prior distributions/{region}_hist_means", m_arr)
    print(f"It took {time.time() - start_time} seconds to roll forward and calculate everything")


def create_hist_dists(region):
    # region: what geographical region to consider

    # Reads rolled forward paths and aggregate them into yearly distributions

    m_arr = np.load(f"Prior distributions/{region}_hist_means.npy")

    # Calculate mean of means and calculate variance from the sample of mean returns
    # Shape: (assets, years, mean/var)
    mv_agg = np.zeros(shape=(m_arr.shape[0], m_arr.shape[1], 2))
    for p in range(mv_agg.shape[0]):
        for y in range(mv_agg.shape[1]):
            # Fill in mean/variance for the mixing distribution per time step
            mv_agg[p, y, 0] = np.mean(m_arr[p, y, :])
            mv_agg[p, y, 1] = np.var(m_arr[p, y, :])

    # Save the numpy arrays in .npy files
    np.save(f"Prior distributions/{region}_hist_dists", mv_agg)
    return mv_agg


def create_prior(region, scale, ratio):
    # region: what geographical region to consider
    # npaths: number of paths to be simulated
    # scale: constant to scale distributions by to create noise
    # ratio: ratio between historical roll forward and noise

    # Adds noisy measurements to empirically determined distributions

    start_time = time.time()
    # Open empirical distributions and paths
    dists = np.load(f"Prior distributions/{region}_hist_dists.npy")
    m_arr = np.load(f"Prior distributions/{region}_hist_means.npy")
    asset_count, npaths = dists.shape[0], m_arr.shape[-1]
    # Calculate lambda to use for mixing the distributions
    l = 1/(1 - ratio)
    for i in range(asset_count):
        # Extract means and variances
        asset = dists[i]
        means = asset[:, 0]
        cov = scale**2 * np.diag(asset[:, 1])
        # Keep RNG consistent
        rng = np.random.default_rng()
        draws = rng.multivariate_normal(means, cov, size=npaths)
        # Add drawn means to means array for path allocation later
        m_arr[i, :, :] += l * draws.transpose()
        # Calculate and store sample mean, sample variance for prior distribution
        dists[i, :, :] += np.array([l * draws.mean(axis=0), l**2 * np.var(draws, axis=0)]).transpose()

    # Save the mixed means, distributions, and used parameters
    np.save(f"Prior distributions/{region}_mixed_means", m_arr)
    np.save(f"Prior distributions/{region}_mixed_dists", dists)
    np.save(f"Prior distributions/{region}_mixing_params", np.array([scale, ratio]))
    print(f"It took {time.time() - start_time} seconds to create the prior")
    return m_arr, dists

def ngfs_predictions(region, horizon):
    # region: what geographical region to consider

    global scenario_list, stocks, models
    scenario_preds = {s: np.zeros(shape=(len(stocks[region]), 2, (horizon - 2026 + 1))) for s in scenario_list}
    for s in scenario_list:
        data = ngfs_pull(region, s)
        v = 0
        for asset in models[region]:
            preds = models[region][asset][0].get_prediction(data)
            scenario_preds[s][v, :, :] = np.exp(preds.predicted_mean), np.square(preds.se)
            v += 1
    return scenario_preds

def assign_paths(region):
    # region: what geographical region to consider

    # Assign each mixed path to an NGFS scenario based on minimized Mahalanobis distance

    global scenario_list, stocks, models
    start_time = time.time()
    # Initiate data
    mixed_paths = np.load(f"Prior distributions/{region}_mixed_means.npy")
    scenario_preds = ngfs_predictions(region, 2050)

    # Keep track of path indices per scenario
    scenario_paths = {s: set() for s in scenario_list}
    npaths = mixed_paths.shape[-1]

    for i in range(npaths):
        # Initialize loop
        best_dist = np.inf
        best_scenario = ""
        for s in scenario_list:
            # Read scenario predictions
            s_data = scenario_preds[s]
            mah_dists = [0] * s_data.shape[0]
            for j in range(len(mah_dists)):
                # Separate means and variance from array and calculate distance
                means, vars = s_data[j, 0, :], np.diag(s_data[j, 1, :])
                mah_dists[j] = mahalanobis(mixed_paths[j, :, i], means, vars)
            cur_dist = np.mean(mah_dists)
            # Sets best scenario if distance is smaller
            if cur_dist <= best_dist:
                best_dist = cur_dist
                best_scenario = s
        # Add index to appropriate set
        scenario_paths[best_scenario].add(i)

    print(f"It took {time.time() - start_time} seconds assign all paths")
    return scenario_paths

def foo():
    # Combine all steps in one
    # Perform optimization
    return


# TODO: Complete step 4, 5 in the mail
# TODO: Perform sensitivity analysis

# %%
roll_macro("US", J, t_end)
# %%
US_roll = create_hist_dists("US")
# %%
prior = create_prior("US", 5, 0.5)
patapim_too = assign_paths("US")
# %%
patapim = ngfs_predictions("US", t_end)
# %%

# %%
dat = ngfs_pull("US", "Delayed transition")