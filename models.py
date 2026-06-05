import sys
import numpy as np
import pandas as pd
from plotnine import *
from plotnine import __version__ as p9__version__
import statsmodels.api as sm
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
    data = pd.read_csv(f'Historical data/{region}/{region}_historical.csv')
    y = pd.read_csv(f'Historical data/{region}/{name}.csv')

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
    variables = ['pr', 'CPI', 'rGDP']
    scalar = StandardScaler()

    # Add additional variables, exclude some to prevent autocorrelation
    for asset_class, var in class_to_var.items():
        if not ac == asset_class:
            variables.append(var)
    if region != 'US':
        variables.append('er')
    # Standardize data if needed
    if std:
        data[variables] = scalar.fit_transform(data[variables])
        data['y'] = scalar.fit_transform(data[['y']])

    formula = f'y ~ {'+'.join(variables)} + 0'
    return smf.ols(formula, data=data).fit()


def predict(model, region, scenario):
    nigem_regions = {'US': 'NiGEM NGFS v1.24.2|United States',
                     'EU': "NiGEM NGFS v1.24.2|Europe"}
    df = pd.read_csv(f'NGFS data/NiGEM_working_set.csv')
    df = df.loc[df['Region'] == nigem_regions[region]]

    if scenario in ['Net Zero 2050', 'Fragmented World', 'Delayed transition']:
        var_type = 'combined'
    else:
        var_type = 'physical'

    reg_variables = [f"Central bank Intervention rate (policy interest rate) ; %({var_type})",
                     f'Equity prices({var_type})',
                     f'Gross Domestic Product (GDP)({var_type})',
                     f'Inflation rate ; %({var_type})',
                     f'Long term interest rate ; %({var_type})']
    df = df[(df["Variable"].isin(reg_variables)
             |
             (df["Variable"].str.contains('Exchange rate;', case=False, regex=True)
              &
              df["Variable"].str.contains(var_type, case=False, regex=True)))
            &
            df["Scenario"].str.contains(scenario, case=False)]


    df.drop(df.iloc[:, 0:4], axis=1, inplace=True)
    df.drop("Unit", axis=1, inplace=True)
    df.set_index("Variable", inplace=True)

    reg_labels = ['pr', 'ep', 'rGDP', 'CPI', 'ltir']
    mapping = {}
    for i in range(0, len(reg_labels)):
        try:
            mapping[reg_labels[i]] = df.loc[reg_variables[i]]
        except KeyError:
            continue

    working_df = pd.DataFrame(mapping)
    predictions = model.predict(working_df)
    return np.exp(predictions)


scenario_list = ["Net Zero 2050", "Delayed transition", "Current Policies", "Fragmented World"]
# US assets
sp500 = create_model('SP500', 'US', ac='index')
dgs10 = create_model('DGS10', 'US', ac='govbond')
bbb = create_model('BAMLC0A4CBBBEY', 'US', ac='index')

sp500_std = create_model('SP500', 'US', True, ac='index')

print(sp500_std.summary())

#Regress on NGFS scenario data
#TODO Clean up code

for sc in scenario_list:
    test = predict(bbb, 'US', 'Net Zero 2050')
#TODO Add mixing for scenario's

