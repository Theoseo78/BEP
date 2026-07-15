# Main file for working with all data
import pandas as pd
import numpy as np
import time
from reg_models import create_model, ngfs_pull
from hmm_training import find_optimal_model
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize
from scipy.special import logsumexp
from copy import copy

# Initialize variables
rs = check_random_state(547362)
scenario_list = [
    "Net Zero 2050",
    "Delayed transition",
    "Current Policies",
    "Fragmented World",
]
# This also determines the order of the assets in the numpy arrays
stocks = {'US': [('SP500', 'index'),
                 ('DGS10', 'govbond'),
                 ('BAMLC0A4CBBBEY', 'index')],
          'EU': [('EURO STOXX 50', 'index'),
                ('MSCI', 'index')],
          }

region_lst = stocks.keys()

# Train regression models
# One for predictions, one for sensitivity analysis
models = {}
for r in stocks.keys():
    models[r] = {}
    for prod in stocks[r]:
        models[r][prod[0]] = create_model(prod[0], r, False, ac=prod[1])

# Finds optimal HMM per region
opt_hmm = {}
for r in region_lst:
    df = pd.read_csv(f"Historical data/{r}_economical_historical.csv")
    df = df.drop(df.columns[0], axis=1)

    # Reshape to (observations, variables)
    data_arr = df.to_numpy().reshape(-1, len(df.columns))
    # Construct HMM model on the data
    opt_hmm[r] = find_optimal_model(data_arr, 10)
    print(f"region: {r}")


# Initialize variables for creating the posterior
J = 20000
t_begin = 2026
t_end = 2050

# %%
def roll_macro(region, npaths, horizon):
    # region: what geographical region to consider
    # npaths: Number of paths to be simulated
    # horizon: up to and including what year should be simulated

    # Creates historical asset paths based on historical data rolled forward from an HMM trained on it

    # Calculate length of each path based on specified horizon
    n_samp = (horizon - 2026 + 1) * 12

    # Create numpy array with size (number of years, number of variables, number of paths)
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
            m_arr[v, :, i] = np.exp(m.predict(df_path))
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

def ngfs_predictions(region, horizon):
    # region: what geographical region to consider

    global scenario_list, stocks, models
    # Assign each scenario its own predictions
    scenario_preds = {s: np.zeros(shape=(len(stocks[region]), (horizon - 2026 + 1))) for s in scenario_list}
    for s in scenario_list:
        data = ngfs_pull(region, s)
        v = 0
        for asset in models[region]:
            preds = models[region][asset].predict(data)
            scenario_preds[s][v, :] = np.exp(preds)
            v += 1
    return scenario_preds

def create_prior(regions, scale, ratio):
    # region: what geographical region to consider
    # npaths: number of paths to be simulated
    # scale: constant to scale distributions by to create noise
    # ratio: ratio between historical roll forward and noise

    # Adds noisy measurements to empirically determined distributions
    # Noise is centered around NGFS scenario with variance equal to (scale * sigma)^2

    start_time = time.time()
    # Open empirical distributions and paths
    for region in regions:
        dists = np.load(f"Prior distributions/{region}_hist_dists.npy")
        m_arr = np.load(f"Prior distributions/{region}_hist_means.npy")
        asset_count, npaths = dists.shape[0], m_arr.shape[-1]
        # Calculate lambda to use for mixing the distributions
        l = ratio/(1 - ratio)
        i = 0
        for asset in stocks[region]:
            # Keep RNG consistent
            rng = np.random.default_rng()
            # Generate noise per scenario
            for s in scenario_list:
                data = ngfs_pull(region, s)
                sc_distribution = models[region][asset[0]].get_prediction(data)
                sc_mean, sc_var = sc_distribution.predicted_mean, np.diag((scale * sc_distribution.se)**2)
                draws = rng.multivariate_normal(sc_mean, sc_var, size=npaths)
                # Add drawn means to means array for path allocation later
                m_arr[i, :, :] += (l/len(scenario_list)) * draws.transpose()
                # Calculate and store sample mean, sample variance for prior distribution
                dists[i, :, :] += np.array([(l/len(scenario_list)) * draws.mean(axis=0), (l/len(scenario_list))**2 * np.var(draws, axis=0)]).transpose()

        # Save the mixed means, distributions, and used parameters
        np.save(f"Prior distributions/{region}_mixed_means", m_arr)
        np.save(f"Prior distributions/{region}_mixed_dists", dists)
        np.save(f"Prior distributions/{region}_mixing_params", np.array([scale, ratio]))
    print(f"It took {time.time() - start_time} seconds to create the prior")

def k_means_paths(regions, horizon):
    # regions: list of economical regions
    start_time = time.time()
    # Assign paths to scenario's by using k-means clustering
    # The centroids are the scenario predictions
    predictions = {r: ngfs_predictions(r, horizon) for r in regions}
    # Concatenate the predictions for all assets per region into one vector for centroids
    pred_vectors = {s: np.concatenate([np.concatenate(predictions[r][s]) for r in regions]) for s in scenario_list}
    pred_matrix = np.array([pred_vectors[s] for s in pred_vectors.keys()])
    paths = np.concatenate([np.concatenate(np.load(f"Prior distributions/{r}_mixed_means.npy")) for r in regions])
    res = KMeans(n_clusters=pred_matrix.shape[0], init=pred_matrix).fit_predict(paths.T)
    scenario_paths = {s: set() for s in scenario_list}
    label_lst = list(res)
    for i in range(len(label_lst)):
        scenario_paths[scenario_list[label_lst[i]]].add(i)
    print(f"It took {time.time() - start_time} seconds to assign all the paths")
    return scenario_paths

# REDUNDANT, only here for historical purposes
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
    # Remove all scenario's without a path
    for s in scenario_list:
        if len(scenario_paths[s]) == 0:
            scenario_paths.pop(s)

    print(f"It took {time.time() - start_time} seconds assign all paths")
    return scenario_paths


def calc_likelihood(regions, s_paths):
    # region: what geographical region to consider

    # Calculates the relative log-likelihood of a scenario happening if it has paths assigned to it
    # Log-likelihood is calculated using the HMM for the respective region
    # Log-likelihood based on macroeconomic variables
    scenario_scores = {s: float() for s in scenario_list}
    for r in regions:
        for s, paths in s_paths.items():
            if len(paths) > 0:
                # Call model to calculate log-likelihood
                s_data = ngfs_pull(r, s)
                s_data_arr = s_data.to_numpy().reshape(-1, len(s_data.columns))
                scenario_scores[s] += opt_hmm[r].score(s_data_arr)
            else:
                continue
    # Rescale log-likelihood when done
    norm_fac = sum(scenario_scores.values())
    for k, v in scenario_scores.items():
        scenario_scores[k] = v / norm_fac

    return scenario_scores

# REDUNDANT, only here for historical purposes
def create_matrices(year, regions, s_paths, s_scores):
    # year: ranges from 2026 to t_end
    # region: list of regions
    # s_paths: dictionary of {scenario: set(path_indices)}
    # s_scores: relative log-likelihood of scenario's

    # Create arrays that form the constraints and target values

    # Create selection array for probability of scenario's
    # Work using scenario's passed from scores
    scenario_lst = s_paths.keys()
    s = len(scenario_lst)
    g_prob = np.zeros((s, J))
    b_prob = np.zeros(s)
    v = 0
    for sc, paths in s_paths.items():
        b_prob[v] = s_scores[sc]
        for p in paths:
            g_prob[v][p] += 1
        v += 1

    # Create array and vector of macroeconomic variables
    g = copy(g_prob)
    b = copy(b_prob)
    if not 2026 <= year <= t_end:
        raise ValueError("Specified year must be at least 2026 up to end of horizon")
    else:
        idx = year - 2026
    for r in regions:
        hist_dat = np.load(f"Prior distributions/{r}_hist_forward.npy")
        hist_dat = hist_dat[idx, :, :]
        v = 0
        for sc in scenario_list:
            if not abs(s_scores[sc]) > 0:
                continue
            # Take means over the years to get characteristic values
            scenario_macro = ngfs_pull(r, sc).to_numpy()[idx]
            # Extend selection matrix to encompass number of variables
            selection = np.tile(g_prob[v], (hist_dat.shape[0], 1))
            g = np.vstack((g, (hist_dat * selection)))
            b = np.concatenate([b, scenario_macro])
            v += 1
    return g, b

def create_general_matrices(regions, s_paths, s_scores, eps):
    # region: list of regions
    # s_paths: dictionary of {scenario: set(path_indices)}
    # s_scores: relative log-likelihood of scenario's
    # eps: amount of error in characteristic values (percent)

    # Create arrays that form the constraints and target values for the year 2050

    # Create selection array for probability of scenario's
    # Work using scenario's passed from scores
    scenario_lst = s_paths.keys()
    s = len(scenario_lst)
    v = 0
    # Create a submatrix of [probability : characteristic values] per scenario and concatenate all
    g, b = None, None
    for sc, paths in s_paths.items():
        g_prob = np.zeros([2, J])
        b_prob = np.array([s_scores[sc], -s_scores[sc]])
        if g is None:
            g = g_prob
            b = b_prob
        else:
            g = np.vstack([g, g_prob])
            b = np.concatenate([b, b_prob])
        for p in paths:
            g_prob[:, p] = [1, -1]
        # Concatenate all regions onto each other in order of regions list
        for r in regions:
            hist_dat = np.load(f"Prior distributions/{r}_hist_forward.npy")[-1, :, :]
            # Multiply element-wise with selection matrix to select the assigned paths
            # Account for lower bound
            g_mean = np.vstack([hist_dat, -1 * hist_dat])
            char_vals = ngfs_pull(r, sc).to_numpy()[-1, :]
            b_mean = np.concatenate([(1 + eps) * char_vals, -(1 - eps) * char_vals])
            # Only have the positive values be the selection matrix
            selection = np.tile(g_prob[0], (2*hist_dat.shape[0], 1))
            g = np.vstack([g, (g_mean * selection)])
            b = np.concatenate([b, b_mean])
        v += 1
    return g, b

def dual(G, b):
    # G: constraints matrix
    # b: constraints value vector

    # Solves Lagrangian dual to retrieve posterior weights

    # Prior weights are all the same
    start_time = time.time()
    p0 = 1/J * np.ones(J)
    # Amount of constraints
    c_num = np.shape(G)[0]
    logZ = lambda theta: logsumexp(np.log(p0) - G.T @ theta)
    f = lambda theta:  b @ theta + logZ(theta)
    # Additional information the solve can use
    theta0 = np.zeros(c_num)
    # Lagrange terms must be greater or equal to zero
    options = {'maxiter': 100, 'maxfun': int(1e6)}
    res = minimize(lambda x:-1*f(x), x0=theta0, method="L-BFGS-B", bounds=[(0, None)] *  c_num, options=options)
    print(f"Success: {res.success}")
    print(f"{res.message}. \nObjective function value: {res.fun}. \nIterations: n = {res.nit}")
    print(f"Largest component in jacobian: {max(abs(res.jac))}")
    theta = res.x
    p_post = np.exp(np.log(p0) - G.T @ theta - logZ(theta))
    # Returns log(p*), where p* are the posterior weights.
    print(f"It took {time.time() - start_time} seconds to complete the optimization")
    return p_post, theta

def create_posterior(t_begin, t_end, weights):
    # weights: vector of posterior weights

    # Creates array containing posterior mean and variance for all assets for all years in the time horizon

    asset_count = sum(len(stocks[r]) for r in stocks.keys())
    post_dists = np.zeros([asset_count, (t_end - t_begin + 1), 2])
    i = 0
    for r in stocks.keys():
        paths = np.load(f"Prior distributions/{r}_mixed_means.npy")
        # Iterate over all assets per region
        for j in range(paths.shape[0]):
            prior_means = paths[j]
            # vector of all posterior means
            mu = prior_means @ weights
            # Calculate variance for weighted mean
            var = (np.square(prior_means - mu) @ weights)/(1 - sum(np.square(weights)))
            post_dists[i,:,0], post_dists[i,:,1] = mu, var
            i += 1
    return post_dists

def create_portfolio(mu, sigma, delta):
    # mu: expected value of assets
    # sigma: variance vector/matrix of assets
    # delta: risk aversion factor: how little risk the investor is willing to take

    ob_func = lambda w, delta: w.T @ mu - (delta/2) * (w.T @ sigma @ w)
    # Minimize function subject to:
    # Weights are in [0, 1], Sum of weights equal to 1
    port_weights = minimize(lambda x: -1 * ob_func(x, delta), [1/len(mu)] * len(mu), bounds=[(0, 1)] * len(mu),
                            constraints=({'type': 'eq', 'fun': lambda x: sum(x)-1}))
    return port_weights.x

def compare_portfolios():
    pass

def glide_path():
    pass

# TODO: Perform sensitivity analysis

# All comments containing the double percent signs are read as an individual cell of code by the PyCharm IDE (and other IDE's)
# This is used to run steps of the model individually, as certain steps (primarily step one) can take a very long time

if __name__ == '__main__':
    # %%
    # Overview of main.py
    # Step one: create macro-economic paths per region
    for region in region_lst:
        roll_macro("EU", J, t_end)
    # %%
    # Step two: create prior distribution by mixing noise with observations
    prior = create_prior(region_lst, 3, 0.15)
    # %%
    # Step three: calculate relative log-likelihood for scenarios, and assign paths to scenarios with k-means clustering
    kmtest = k_means_paths(region_lst, t_end)
    kmscores = calc_likelihood(region_lst, kmtest)
    # %%
    # Step four: solve dual optimization problem and get the posterior weights
    p_star, theta = dual(*create_general_matrices(stocks.keys(),kmtest , kmscores, 0.20))
    # %%
    # Step five: Compute the posterior distribution for the end of the time horizon
    post_dists = create_posterior(t_begin, t_end, p_star)
    # %%
    # Step six: Create portfolio based on target year and given risk aversion
    mu, sigma = post_dists[:, -1, 0], np.diag(post_dists[:, -1, 1])
    delta = 2
    portfolio = create_portfolio(mu, sigma, delta)


    # Lines for debugging
    # # %%
    # p_starr, thetar = dual(*create_matrices(t_end, ["US"], kmtest, kmscores))
    # # %%
    # c_matr, target = create_general_matrices(stocks.keys(),kmtest, kmscores, 0)
    # c_matr = c_matr.T
    # # %%
    # g_mean_test = create_general_matrices(["US"],kmtest, kmscores, 0)
    # # %%
    # view = np.load((f"Prior distributions/EU_hist_forward.npy"))