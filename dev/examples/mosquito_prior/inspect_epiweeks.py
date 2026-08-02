import pandas as pd

df = pd.read_excel(r'C:\Users\shaf0043\Desktop\Mosquito_reserve\Mosquito_Data_Merged_with_Cov_Data_clean.xlsx')

# Apply the same filter as our run: keep years with >= 20 unique epiweeks
valid_years = [yr for yr in sorted(df['Year'].unique())
               if df[df['Year'] == yr]['Epiweek'].nunique() >= 20]
df = df[df['Year'].isin(valid_years)]

epiweeks  = list(range(int(df['Epiweek'].min()), int(df['Epiweek'].max()) + 1))
years     = valid_years
locations = sorted(df['Location.Name'].unique())

n_ew   = len(epiweeks)
n_yr   = len(years)
n_loc  = len(locations)
total  = n_loc * n_yr * n_ew   # total possible (location, year, epiweek) cells

# Count unique (location, year, epiweek) triplets that actually exist
present = df.groupby(['Location.Name', 'Year', 'Epiweek']).ngroups
missing = total - present

print(f'Grid definition')
print(f'  Years     : {n_yr}  ({years[0]} – {years[-1]})')
print(f'  Epiweeks  : {n_ew}  (labels {epiweeks[0]} – {epiweeks[-1]})')
print(f'  Locations : {n_loc}')
print(f'  Total possible cells : {n_loc} × {n_yr} × {n_ew} = {total:,}')
print()
print(f'  Valid (observed) cells : {present:,}   ({present/total*100:.1f}%)')
print(f'  Missing (zero-padded)  : {missing:,}   ({missing/total*100:.1f}%)')
print()
print(f'Per-year breakdown (across all {n_loc} locations × {n_ew} epiweeks = {n_loc*n_ew} cells/year):')
print(f'  {"Year":>6}  {"Valid":>8}  {"Missing":>8}  {"Valid%":>7}')
print(f'  {"-"*6}  {"-"*8}  {"-"*8}  {"-"*7}')
for yr in years:
    yr_present = df[df['Year'] == yr].groupby(['Location.Name', 'Epiweek']).ngroups
    yr_total   = n_loc * n_ew
    yr_missing = yr_total - yr_present
    print(f'  {yr:>6}  {yr_present:>8,}  {yr_missing:>8,}  {yr_present/yr_total*100:>6.1f}%')
