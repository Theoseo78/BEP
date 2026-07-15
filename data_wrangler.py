# Python file used to wrangle data fit for hmm training
# See hmm_training.py for file specifications

# %%
import pandas as pd
from functools import reduce
import numpy as np
# %%
# US inflation rate
df = pd.read_csv(r"Historical data/PCE_unedited.csv", header=0)
df = df.transpose()
dates = pd.date_range(start='2021-01-01', freq="MS", periods=60)
new_df = pd.DataFrame(columns=["observation_date", "PCE"])
new_df["observation_date"] = dates
new_df["PCE"] = df.iloc[:, -1][1:].to_numpy()
new_df.to_csv(r"Historical data/US/PCE.csv", index = False)
# %%
# Surface temperature
# https://wmo.int/resources/dashboards/global-mean-temperature-1850-2024
df = pd.read_csv(r"Historical data/hmm training/mean_temp.csv", index_col="year")
new_df = pd.DataFrame(df.mean(axis=1))
new_df.to_csv(r"Historical data/hmm training/mean_surface_temp.csv", index = False)
# %%
# Primary energy mix
# https://ourworldindata.org/energy-mix
df = pd.read_csv(r"Historical data/hmm training/global-energy-substitution.csv", index_col="Year")
df = df["World"]
df.drop(columns=["Entity", "Code"], inplace=True)
# Primary energy from fossil fuels/All primary energy
df["Mix"] = df.iloc[:, -4:].sum(axis=1)/ df.iloc[:, 0:10].sum(axis=1)
new_df = pd.DataFrame(df["Mix"])
new_df.to_csv(r"Historical data/hmm training/historical_pem.csv", index = False)
# %%
# Wrangle IAM data to account for primary energy mix
df = pd.read_csv(r"NGFS data/IAM_working_set.csv")
variables = ["Primary Energy", "Primary Energy|Fossil"]
temp = df[df["Variable"].isin(variables) & df["Region"].str.contains("World")]

def ratio(series):
    return reduce(lambda x, y: y / x, series)
temp = temp.drop(["Model", "Region", "Unit", "Unnamed: 0"], axis=1)
temp = temp.groupby(["Scenario", "Variable"], as_index=True).mean()
emp = temp.groupby(["Scenario"], as_index=True).agg(ratio)
emp = emp.interpolate(axis=1)
emp.to_csv(r"Historical data/hmm training/pem_iam.csv", index = True)
# %%
# CO2 emissions
# https://ourworldindata.org/co2-emissions
df = pd.read_csv(r"Historical data/hmm training/annual-co2-emissions-per-country.csv", index_col="Year")
df.drop(columns=["Entity", "Code"], inplace=True)
df = df/10e6
df.to_csv(r"Historical data/hmm training/historical_ce.csv", index = True)
# %%
# EU long term interest rate
df = pd.read_csv(r"Historical data/EU/Macro economic data/ECB_pr_unedited.csv")
df.ffill(inplace=True)
df.to_csv(r"Historical data/EU/Macro economic data/ECB_pr.csv")
# %%
# EU to USD
df = pd.read_csv(r"Historical data/EU/Macro economic data/ECB_er_unedited.csv")
df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: x/1.1836)
df.to_csv(r"Historical data/EU/Macro economic data/ECB_er.csv")
# %%
# Convert MSCI corporate bond data from excel to .csv
df = pd.read_excel("733216 - MSCI EUR High Yield Selection Corporate Bond Index - 2021-01-01 - 2025-12-31 - Monthly.xlsx")
df.to_csv(r"Historical data/EU/Stock data/MSCI.csv", index = True)
# %%
df = pd.read_csv("Historical data/EU/Stock data/MSCI.csv")