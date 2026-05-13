import sys
import numpy as np
import pandas as pd
from plotnine import *
from plotnine import __version__ as p9__version__
import statsmodels.api as sm
import statsmodels.formula.api as smf

def create_model(name, region, **kwargs):
    ac = kwargs.pop("ac", None)
    class_to_var = {'index':  'epi',
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
    print(y)

    # Work with log prices
    formula = 'np.log(y) ~ pr + CPI + rGDP'
    for asset_class, var in class_to_var.items():
        if not ac == asset_class:
            formula += ' + ' + var
    if region != 'US':
        formula += ' + er'
    return smf.ols(formula, data=data).fit()

#TODO Add regression models for other asset classes
sp500 = create_model('SP500', 'US', ac='index')
dgs10 = create_model('DGS10', 'US', ac='govbond')
bbb = create_model('BAMLC0A4CBBBEY', 'US', ac='index')



#TODO Regress on NGFS scenario data