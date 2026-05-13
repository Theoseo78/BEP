import pyam

conn = pyam.iiasa.Connection("ngfs_phase_5")

# print(conn.scenarios())  # confirm exact scenario name strings
# print(conn.models())  # confirm exact model strings
# print(conn.regions())  # confirm region name strings

vars_df = conn.variables()

# search for the macro-financial variables you need
print(vars_df[vars_df.str.contains("Equity", case=False)])
print(vars_df[vars_df.str.contains("Interest Rate", case=False)])
print(vars_df[vars_df.str.contains("Inflation", case=False)])
print(vars_df[vars_df.str.contains("Exchange Rate", case=False)])
print(vars_df[vars_df.str.contains("GDP", case=False)])
print(vars_df[vars_df.str.contains("Carbon", case=False)])
