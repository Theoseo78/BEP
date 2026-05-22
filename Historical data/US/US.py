import pandas as pd
import sys

# Time span of data: 5 years of monthly data

# Pulling data for equity price index, Dow Jones data
# Also contains the observation date (YYYY - MM - DD)
# Index 2017 = 100, percentage change compared to 2022
epi = pd.read_csv("DJIA_NBD20220101.csv")
epi.iloc[:, 1] = epi.iloc[:, 1] - 100
epi.rename({"DJIA_NBD20220101": "epi"}, axis=1, inplace=True)


# Long-term interest rate
# Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10)
#TODO Use different time series for long term interest rate
ltir = pd.read_csv("IRLTLT01USM156N_CHG.csv")
ltir.rename({"IRLTLT01USM156N_CHG": "ltir"}, axis=1, inplace=True)
ltir = ltir[["ltir"]]


# Pulling data for policy rates from BIS
pr = pd.read_csv("FEDFUNDS.csv")
pr = pr[['FEDFUNDS']]
pr.rename({'FEDFUNDS': 'pr'}, axis=1, inplace=True)

# CPI/inflation
# Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)
cpi = pd.read_csv("CPIAUCSL_PCH.csv")
cpi = cpi[["CPIAUCSL_PCH"]].iloc[:-1]
cpi.rename({"CPIAUCSL_PCH": "CPI"}, axis=1, inplace=True)

# Real GDP
# Real Gross Domestic Product (GDPC1)
gdp = pd.read_csv("GDPC1.csv")
# Convert quarterly measurements into interpolating monthly
new_gdp = [0] * (len(cpi) + 1)
old_gdp = list(gdp["GDPC1"])
# Apply linear interpolation
for i in range(0, len(old_gdp) - 1):
    new_gdp[3 * i], new_gdp[3 * i + 1], new_gdp[3 * i + 2], new_gdp[3 * (i + 1)] = (
        old_gdp[i],
        (2/3) * old_gdp[i] + (1/3) * old_gdp[i + 1],
        (1/3) * old_gdp[i] + (2/3) * old_gdp[i + 1],
        old_gdp[i + 1],
    )
# Drop last observation to keep only 60 measurements
new_gdp = new_gdp[:-1]
gdp = pd.DataFrame(new_gdp, columns=["rGDP"])

# Concatenating all dataframes into one big frame for regression
Master = pd.concat([epi, ltir, pr, cpi, gdp], axis=1)
Master.to_csv('US_historical.csv')