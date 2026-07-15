import sys
import numpy as np
import pandas as pd
# from plotnine import *
# from plotnine import __version__ as p9__version__
# import statsmodels.api as sm
from hmm_training import calculate_mix_weights
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# scenario's
# Net Zero 2050
# Delayed transition
# Current Policies
# Fragmented World


def create_model(name, region, std=False, **kwargs):
    ac = kwargs.pop("ac", None)
    class_to_var = {'index': 'epi',
                    'govbond': 'ltir'}
    # Reading the data
    data = pd.read_csv(f'Historical data/{region}_economical_historical.csv')
    y = pd.read_csv(f'Historical data/{region}/Stock data/{name}.csv')

    # Correct if there is mismatch in size
    if len(y) < len(data):
        # Shift existing index downward
        offset = len(data) - len(y)
        y.index = y.index + offset

        # Reindex to full size
        y = y.reindex(range(len(data)))

    data.insert(2, 'y', y[f'{name}'])
    # Work with log prices
    data['y'] = np.log(data['y'])
    variables = ['pr', 'CPI']
    scalar = StandardScaler()

    # Add additional variables, exclude some to prevent autocorrelation
    for asset_class, var in class_to_var.items():
        if not ac == asset_class:
            variables.append(var)
    if region != 'US':
        variables.append('er')
    if region != 'EU':
        variables.append('rGDP')
    # Standardize data if needed
    if std:
        data[variables] = scalar.fit_transform(data[variables])
        data['y'] = scalar.fit_transform(data[['y']])

    formula = f'y ~ {' + '.join(variables)} '
    return smf.ols(formula, data=data).fit()


def ngfs_pull(region, scenario):
    # Map region string to NiGEM regions
    nigem_regions = {'US': 'NiGEM NGFS v1.24.2|United States',
                     'EU': "NiGEM NGFS v1.24.2|Europe"}
    region_to_cur = {'US': "",
                     'EU': "Euro"}
    cur = region_to_cur[region]
    df = pd.read_csv(f'NGFS data/NiGEM_working_set.csv')
    df = df.loc[df['Region'] == nigem_regions[region]]


    # Scenario "Current Policies" only works for physical data, no transition data
    if scenario in ['Net Zero 2050', 'Fragmented World', 'Delayed transition']:
        var_type = 'combined'
    else:
        var_type = 'physical'

    reg_variables = [f"Central bank Intervention rate (policy interest rate) ; %({var_type})",
                     f'Equity prices({var_type})',
                     f'Exchange rate; {cur} per US$({var_type})',
                     f'Gross Domestic Product (GDP)({var_type})',
                     f'Inflation rate ; %({var_type})',
                     f'Long term interest rate ; %({var_type})']
    df = df[((df["Variable"].isin(reg_variables)
              &
              df["Scenario"].str.contains(scenario, case=False, regex=True)
              &
              df["Variable"].str.contains(var_type, case=False, regex=True)
              )
             |
             (df["Variable"].str.contains('Exchange rate;', case=False, regex=True)
              &
              df["Variable"].str.contains(var_type, case=False, regex=True)
              &
              df["Scenario"].str.contains(scenario, case=False, regex=True))
             )]

    # Drop unnecessary columns
    df.drop(df.iloc[:, 0:4], axis=1, inplace=True)
    df.drop("Unit", axis=1, inplace=True)
    df.set_index("Variable", inplace=True)

    reg_labels = ['pr', 'epi', 'er', 'rGDP', 'CPI', 'ltir']
    mapping = {}
    for i in range(0, len(reg_labels)):
        try:
            mapping[reg_labels[i]] = df.loc[reg_variables[i]]
        except KeyError:
            continue

    working_df = pd.DataFrame(mapping)
    # Have only observations from year 2026+
    return working_df.iloc[4:]


if __name__ == '__main__':
    test = ngfs_pull("EU", "Current Policies")


    # scenario_list = ["Net Zero 2050", "Delayed transition", "Current Policies", "Fragmented World"]
    # # US assets
    # sp500 = create_model('SP500', 'US', ac='index')
    # dgs10 = create_model('DGS10', 'US', ac='govbond')
    # bbb = create_model('BAMLC0A4CBBBEY', 'US', ac='index')
    # #
    # print(sp500.summary())
    # print(dgs10.summary())
    # print(bbb.summary())
    # t = create_model('MSCI', 'EU', ac='index')
    # print(t.summary())
    #
    # data = pd.read_csv('Historical data/US/US_historical.csv')
    # test = predict(sp500, 'US', scenario_list[0])
    # sp500_std = create_model('SP500', 'US', True, ac='index')
    #
    # print(sp500_std.summary())

    #Regress on NGFS scenario data

    # %%
    # i = 0
    # preds = [None] * len(scenario_list)
    # weights = calculate_mix_weights()
    # for sc in scenario_list:
    #     preds[i] = weights[sc] * predict(sp500, "US", sc)
    #     i += 1
    # total = sum(preds)
    #

