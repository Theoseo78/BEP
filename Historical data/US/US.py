import pandas as pd

# Time span of data: 5 years of monthly data

# Pulling data for equity price index,  Dow Jones data
# Also contains the observation date (YYYY - MM - DD)
epi = pd.read_csv("DJIA.csv")
epi.rename({"DJIA": "epi"}, axis=1, inplace=True)

# Long-term interest rate
# Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10)
#TODO Use different time series for long term interest rate
ltir = pd.read_csv("IRLTLT01USM156N.csv")
ltir.rename({"IRLTLT01USM156N": "ltir"}, axis=1, inplace=True)
ltir = ltir[["ltir"]]

# Pulling data for policy rates from BIS
urls = [f"https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.US?startPeriod=2021-01-01&endPeriod=2026-01-01&format=csv"]
pr = pd.concat([pd.read_csv(url) for url in urls])
pr = pr[['OBS_VALUE']]
pr.rename({'OBS_VALUE': 'pr'}, axis=1, inplace=True)

# CPI/inflation
# Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)
cpi = pd.read_csv("CPIAUCSL_PCH.csv")
cpi = cpi[['CPIAUCSL_PCH']]
cpi.rename({'CPIAUCSL_PCH': 'CPI'}, axis=1, inplace=True)

# Real GDP
# Real Gross Domestic Product (GDPC1)
gdp = pd.read_csv("GDPC1.csv")
# Convert quarterly measurements into interpolating monthly
new_gdp = [0] * len(cpi)
old_gdp = list(gdp['GDPC1'])
for i in range(0, len(old_gdp)):
    new_gdp[3 * i], new_gdp[3 * i + 1], new_gdp[3 * i + 2] = old_gdp[i], old_gdp[i], old_gdp[i]
gdp = pd.DataFrame(new_gdp, columns=['rGDP'])

# Concatenating all dataframes into one big frame for regression
Master = pd.concat([epi, ltir, pr, cpi, gdp], axis=1)
Master = Master.iloc[:-1]
Master.to_csv('US_historical.csv')