# Pulling working data from NGFS data
import pandas as pd


# scenario_list = [
#     "Net Zero 2050",
#     "Delayed transition",
#     "Current Policies",
#     "Fragmented World",
# ]
#
# # NiGEM data
# nigem_regions = [
#     "NiGEM NGFS v1.24.2|United States",
#     "NiGEM NGFS v1.24.2|Japan",
#     "NiGEM NGFS v1.24.2|United Kingdom",
#     "NiGEM NGFS v1.24.2|World",
#     "NiGEM NGFS v1.24.2|Europe",
#     "NiGEM NGFS v1.24.2|China",
# ]
# nigem_var_list = ["Equity", "Interest Rate", "Inflation", "GDP", "Exchange"]
# nigem_vars_join = "|".join(nigem_var_list)
#
# df = pd.read_excel("NiGEM_data.xlsx")
# working_nigem = df[
#     (
#         df["Model"].str.contains("REMIND-MAgPIE 3.3-4.8", case=False)
#         & df["Region"].isin(nigem_regions)
#         & df["Variable"].str.contains(nigem_vars_join, case=False)
#         & df["Scenario"].isin(scenario_list)
#     )
# ]
# working_nigem.to_csv("NiGEM_working_set.csv")


# IAM data
model = "REMIND-MAgPIE 3.3-4.8"
iam_regions = ["United", "Japan", "EU 28", "China", "World"]
iam_regions_join = "|".join(iam_regions)
iam_var_list = ["Emissions|C02", "Price|Carbon"]
scenario_list = [
    "Net Zero 2050",
    "Delayed transition",
    "Current Policies",
    "Fragmented World",
]

df = pd.read_excel("IAM_data.xlsx")
working_iam = df[
    (df["Scenario"].isin(scenario_list) 
            &
            df["Model"].str.contains(model)
            &
            df["Region"].str.contains(iam_regions_join, case=False)
            &
            (df["Variable"].isin(iam_var_list)
             |
             df["Variable"].str.contains("^Primary Energy", regex=True)
             |
             df["Variable"].str.contains("Temperature", case=False))

    )
]
working_iam.to_csv("IAM_working_set.csv")
