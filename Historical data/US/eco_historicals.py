# %%
import numpy as np
import pandas as pd
import sys

regional_files = {"US": {"DJIA_NBD20171201": "epi",
                         "IRLTLT01USM156N_CHG": "ltir",
                         "FEDFUNDS": "pr",
                         "CPIAUCSL_PCH": "CPI",
                         "GDPC1": "rGDP"},
                  }

#TODO: fix the data so that they are in the same order of magnitude of 2022
# Using annual data might be the way to go

# Time span of data: 5 years of monthly data
# File to wrangle US data

# Data from FRED, all data sets have an "observation_date" column

# Pulling data for equity price index, Dow Jones data
# Also contains the observation date (YYYY - MM - DD)

# Index 2017 = 100, percentage change compared to 2022
# Equity price index
# Dow Jones
# %%
#TODO Write function that creats {region}_historical.csv using mapping as argument.
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
        df = pd.read_csv(f'Historical data/{region}/' + file_name + '.csv')
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
        base = df[df.observation_date == "2022-12-01"].to_numpy()
        base = base[0][-1]
        # Apply appropriate mapping for
        if vname in abdiff:
            df[vname] = df[file_name] - base
        elif vname in pdiff:
            df[vname] = (df[file_name] / base - 1) * 100
        else:
            raise ValueError("Variable not in list of available variables")
        cols[k] = df.iloc[:, -1].to_numpy()
        k += 1
    data = np.array(cols).T
    historical = pd.DataFrame(data, columns=variables, index=dates_lst)
    # historical.to_csv(f'Historical data/{region}_economical_historical.csv')
    return historical


t = create_historical("US")

# %%
# epi = pd.read_csv("DJIA_NBD20220101")
# epi.iloc[:, 1] = epi.iloc[:, 1] - 100
# epi.rename({"DJIA_NBD20220101": "epi"}, axis=1, inplace=True)
#
# # Long-term interest rate
# # Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10)
# ltir = pd.read_csv("IRLTLT01USM156N_CHG.csv")
# ltir.rename({"IRLTLT01USM156N_CHG": "ltir"}, axis=1, inplace=True)
# ltir = ltir[["ltir"]]
#
# # Pulling data for policy rates from BIS
# pr = pd.read_csv("FEDFUNDS.csv")
# pr = pr[['FEDFUNDS']]
# pr.rename({'FEDFUNDS': 'pr'}, axis=1, inplace=True)
#
# # CPI/inflation
# # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)
# cpi = pd.read_csv("CPIAUCSL_PCH.csv")
# cpi = cpi[["CPIAUCSL_PCH"]].iloc[:-1]
# cpi.rename({"CPIAUCSL_PCH": "CPI"}, axis=1, inplace=True)
#
# # Real GDP
# # Real Gross Domestic Product (GDPC1)
# gdp = pd.read_csv("GDPC1.csv")
# # Convert quarterly measurements into interpolating monthly
# new_gdp = [0] * (len(cpi) + 1)
# old_gdp = list(gdp["GDPC1"])
# # Apply linear interpolation
# for i in range(0, len(old_gdp) - 1):
#     new_gdp[3 * i], new_gdp[3 * i + 1], new_gdp[3 * i + 2], new_gdp[3 * (i + 1)] = (
#         old_gdp[i],
#         (2 / 3) * old_gdp[i] + (1 / 3) * old_gdp[i + 1],
#         (1 / 3) * old_gdp[i] + (2 / 3) * old_gdp[i + 1],
#         old_gdp[i + 1],
#     )
# # Drop last observation to keep only 60 measurements
# new_gdp = new_gdp[:-1]
# gdp = pd.DataFrame(new_gdp, columns=["rGDP"])
#
# # Concatenating all dataframes into one big frame for regression
# Master = pd.concat([epi, ltir, pr, cpi, gdp], axis=1)
# Master.to_csv('US_historical.csv')
