import numpy as np
import pandas as pd
import sys

# Correct order of variables: ['pr', 'epi', 'er' 'rGDP', 'CPI', 'ltir']

regional_files = {"US": {"FEDFUNDS": "pr",
                         "DJIA_NBD20171201": "epi",
                         "GDPC1": "rGDP",
                         "PCE": "CPI",
                         "IRLTLT01USM156N": "ltir",
                         },
                  "EU": {"ECB_pr": "pr",
                         "ECB_er": "er",
                         "ECB_cpi": "CPI",
                         "ECB_ltir": "ltir",
                         }
    ,}

#TODO: fix the data so that they are in the same order of magnitude of 2022
# Using annual data might be the way to go

# Time span of data: 5 years of monthly data
# File to wrangle US data

# Data from FRED, all data sets have an "observation_date" column

# Pulling data for equity price index, Dow Jones data
# Also contains the observation date (YYYY - MM - DD)

# See "notes on data.txt" for extra notes

# LAST COLUMN MUST BE THE COLUMN CONTAINING THE OBSERVATIONS

def create_historical(region):
    # Read csv
    # Apply correct mapping
    # Apply correct naming
    # Remove redundant columns

    # Initialize variables
    variables = list(regional_files[region].values())
    cols = [None] * len(variables)
    k = 0
    dates_lst = pd.date_range(start='2021-01-01', freq="MS", periods=60)
    for file_name, vname in regional_files[region].items():
        # Map file to correct variable name
        df = pd.read_csv(f'Historical data/{region}/Macro economic data/' + file_name + '.csv')

        # Interpolate dataframe if it doesn't contain 60 entries (60 months over 5 years)
        if len(df) < 60:
            new_df = pd.DataFrame(columns=["observation_date", file_name], index=range(60), dtype=float)
            new_df["observation_date"] = dates_lst
            fac = int(np.ceil(len(new_df) / len(df)))
            for i in range(len(df)):
                new_df.iloc[fac * i, -1] = df.iloc[i, -1]
            df = new_df.interpolate()

        # Interpolate if dataframe contains NaNs
        if df.isnull().values.any():
            df.iloc[:, -1] = df.iloc[:, -1].interpolate()

       # Abs diff: pr, CPI, ltir,
        abdiff = ["pr", "CPI", "ltir"]
        # % diff: epi, er (US data set has none), gdp,
        pdiff = ["epi", "er", "rGDP"]
        # In NiGEM data, 2022 is considered base year
        # 2022-12-01 as "end of period" aggregation
        # Row 23 corresponds to that date
        base = df.iloc[23, -1]

        # Apply appropriate mapping for
        if vname in abdiff:
            df[vname] = df.iloc[:, -1] - base
        elif vname in pdiff:
            df[vname] = (df.iloc[:, -1] / base - 1) * 100
        else:
            raise ValueError("Variable not in list of available variables")
        cols[k] = df.iloc[:, -1].to_numpy()
        k += 1
    data = np.array(cols).T
    historical = pd.DataFrame(data, columns=variables, index=dates_lst)
    historical.to_csv(f'Historical data/{region}_economical_historical.csv')
    return historical
if __name__ == "__main__":
    t = create_historical("EU")




