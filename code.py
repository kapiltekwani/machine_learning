from pandas import DataFrame, read_csv;

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
pd.options.display.float_format = '{:.0f}'.format

print("******* Checkpoint #1 *****")

round2_df = pd.read_csv('rounds2.csv', engine='python', encoding = "palmos")
round2_df['permalink'] = round2_df.company_permalink.str.upper()

print("total number of unique companies in round2.csv")
print (len(round2_df.permalink.unique()))


companies_df = pd.read_csv('companies.txt', sep="\t", encoding = "palmos")
companies_df.permalink = companies_df.permalink.str.upper()

print("Total number of unique records in Companies.txt")
print(len(companies_df.permalink.unique()))

print("Difference of companies in Companies.txt and round2 ")
print(len(companies_df.permalink.unique()) - len(round2_df.permalink.unique()))

print("Count of observations in master frame")
master_frame = pd.merge(round2_df, companies_df, how="inner", on="permalink")
print(len(master_frame))


print("******* Checkpoint #2 *****")
print("Number of rows where 'raised_amount_usd' is NULL")
print(master_frame.raised_amount_usd.isnull().sum())

master_frame.loc[np.isnan(master_frame['raised_amount_usd']), ['raised_amount_usd']] = 0;

grouped_master_frame = master_frame.groupby('funding_round_type')['raised_amount_usd'].mean()

average_funding = pd.DataFrame({'funding_round_type': grouped_master_frame.index, 'raised_amount_usd': grouped_master_frame.values})

print("Most representative value of the investment amount for venture, angel, seed, and private equity funding types")
print(average_funding[average_funding.funding_round_type.isin(['venture','angel', 'seed', 'private_equity'])])

print("Which investment is most suitable for Spark Investments")
print(average_funding[(average_funding['raised_amount_usd'] >= 5000000.00) &
                           (average_funding['raised_amount_usd'] <= 15000000.00) &
                          (average_funding.funding_round_type.isin(['venture','angel', 'seed', 'private_equity']))])


print("******* Checkpoint #3 *****")

print("nine countries which have received the highest total funding (across ALL sectors for the chosen investment type)")
venture_master_frame = master_frame[master_frame.funding_round_type == "venture"]
groupby_country_frame = venture_master_frame.groupby('country_code')['raised_amount_usd'].sum();
top9 = pd.DataFrame({'country_code': groupby_country_frame.index, 'raised_amount_usd':groupby_country_frame.values}).sort_values(by='raised_amount_usd', ascending=False)[0:9]
print(top9)


print("******* Checkpoint #4 *****")
master_frame['primary_sector'] = master_frame.category_list.str.split("|").str.get(0)
print("Mapping each category to one of the eight main sectors")
mapping_df = pd.read_csv('/Users/tkapil/Learning/mapping.csv', engine='python', encoding = "palmos")
mapping_df = mapping_df.set_index('category_list').idxmax(axis=1)
mapping_df = pd.DataFrame({'primary_sector': mapping_df.index, 'main_sector': mapping_df.values})

print(mapping_df)



print("******* Checkpoint #5 *****")

master_frame = pd.merge(master_frame, mapping_df, how="inner", on="primary_sector")

venture_frame = master_frame[(master_frame.funding_round_type == "venture") &
                             (master_frame['raised_amount_usd'] >= 5000000.00) &
                             (master_frame['raised_amount_usd'] <= 15000000.00)]


d1 =  venture_frame[venture_frame.country_code == "USA"]
d2 =  venture_frame[venture_frame.country_code == "GBR"]
d3 =  venture_frame[venture_frame.country_code == "IND"]

print("Total Number of Investments in each country")
print("D1 :", len(d1))
print("D2 :", len(d2))
print("D3 :", len(d3))

print("Total amount of investment in each country ")
print("D1 :", d1.raised_amount_usd.sum())
print("D2 :", d2.raised_amount_usd.sum())
print("D3 :", d3.raised_amount_usd.sum())

print("Top sector based on count of investments")
country_1_investment_count_series = d1.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
country_2_investment_count_series = d2.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)
country_3_investment_count_series = d3.groupby('main_sector')['raised_amount_usd'].count().sort_values(ascending=False)

country_1_investment_count_df = pd.DataFrame(country_1_investment_count_series)
country_2_investment_count_df = pd.DataFrame(country_2_investment_count_series)
country_3_investment_count_df = pd.DataFrame(country_3_investment_count_series)

country_1_investment_count_df.reset_index(inplace=True)
country_2_investment_count_df.reset_index(inplace=True)
country_3_investment_count_df.reset_index(inplace=True)

country_1_investment_count_df['country_code'] = "USA"
country_2_investment_count_df['country_code'] = "GBR"
country_3_investment_count_df['country_code'] = "IND"

country_1_first_main_sector = country_1_investment_count_df.loc[0].main_sector
country_2_first_main_sector = country_2_investment_count_df.loc[0].main_sector
country_3_first_main_sector = country_3_investment_count_df.loc[0].main_sector

country_1_second_main_sector = country_1_investment_count_df.loc[1].main_sector
country_2_second_main_sector = country_2_investment_count_df.loc[1].main_sector
country_3_second_main_sector = country_3_investment_count_df.loc[1].main_sector

country_1_third_main_sector = country_1_investment_count_df.loc[2].main_sector
country_2_third_main_sector = country_2_investment_count_df.loc[2].main_sector
country_3_third_main_sector = country_3_investment_count_df.loc[2].main_sector

print("for D1:", country_1_first_main_sector)
print("for D2:", country_2_first_main_sector)
print("for D3:", country_3_first_main_sector)

print("Second-best sector (based on count of investments)")
print("for D1:", country_1_second_main_sector)
print("for D2:", country_2_second_main_sector)
print("for D3:", country_3_second_main_sector)

print("Third-best sector (based on count of investments")
print("for D1:", country_1_third_main_sector)
print("for D2:", country_2_third_main_sector)
print("for D3:", country_3_third_main_sector)


print("Number of investments in the top sector")
print("for D1:", country_1_investment_count_df.loc[0].raised_amount_usd)
print("for D2:", country_2_investment_count_df.loc[0].raised_amount_usd)
print("for D3:", country_3_investment_count_df.loc[0].raised_amount_usd)

print("Number of investments in the second-best sector")
print("for D1:", country_1_investment_count_df.loc[1].raised_amount_usd)
print("for D2:", country_2_investment_count_df.loc[1].raised_amount_usd)
print("for D3:", country_3_investment_count_df.loc[1].raised_amount_usd)


print("Number of investments in the third-best sector")
print("for D1:", country_1_investment_count_df.loc[2].raised_amount_usd)
print("for D2:", country_2_investment_count_df.loc[2].raised_amount_usd)
print("for D3:", country_3_investment_count_df.loc[2].raised_amount_usd)



print("For the top sector count-wise (point 3), which company received the highest investment?")
print("for D1:", d1[d1.main_sector == country_1_first_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))
print("for D2:", d2[d2.main_sector == country_2_first_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))
print("for D3:", d3[d3.main_sector == country_3_first_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))

print("For the second-best sector count-wise (point 4), which company received the highest investment?")
print("for D1:", d1[d1.main_sector == country_1_second_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))
print("for D2:", d2[d2.main_sector == country_2_second_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))
print("for D3:", d3[d3.main_sector == country_3_second_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))

print("Number of investments in the third-best sector")
print("for D1:", d1[d1.main_sector == country_1_third_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))
print("for D2:", d2[d2.main_sector == country_2_third_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))
print("for D3:", d3[d3.main_sector == country_3_third_main_sector].groupby('permalink')[['raised_amount_usd']].sum().sort_values(by='raised_amount_usd', ascending=False).head(1))


print("******* Checkpoint #6 *****")

average_funding = average_funding[average_funding.funding_round_type.isin(['venture','angel', 'seed', 'private_equity'])]

grouped_master_frame = master_frame.groupby('funding_round_type')['raised_amount_usd'].sum()
total_funding = pd.DataFrame({'funding_round_type': grouped_master_frame.index, 'raised_amount_usd': grouped_master_frame.values})
total_funding = total_funding[total_funding.funding_round_type.isin(['venture','angel', 'seed', 'private_equity'])]

plt.figure(figsize = (12,8))

plt.subplot(1,2,1)
plt.title("Average Investent - Funding type wise")
plt.yscale('log')
sns.barplot(x='funding_round_type', y = "raised_amount_usd", data = average_funding)
plt.xlabel("Funding Type")
plt.ylabel("Average Investment Amount")

plt.subplot(1,2,2)
plt.title("Total Investent - Funding type wise")
plt.yscale('log')
sns.barplot(x='funding_round_type', y = "raised_amount_usd", data = total_funding)
plt.xlabel("Funding Type")
plt.ylabel("Total Investment Amount")
plt.show();

print("Plot showing the top 9 countries against the total amount of investments of funding type venture")
plt.figure(figsize = (12,8))
plt.title("Total Investent - Country wise")
sns.barplot(x='country_code', y = "raised_amount_usd", data = top9)
plt.xlabel("Country Code")
plt.ylabel("Total Investment Amount")
plt.show();


country_sector_wise_count_frame = country_1_investment_count_df.head(3)
country_sector_wise_count_frame = country_sector_wise_count_frame.append(country_2_investment_count_df.head(3))
country_sector_wise_count_frame = country_sector_wise_count_frame.append(country_3_investment_count_df.head(3))
plt.figure(num = None, figsize = (12,8), dpi = 80, facecolor='w', edgecolor='k')
plt.title("Number of investments in the top 3 sectors of the top 3 countries")
sns.barplot(x='country_code', y = "raised_amount_usd", hue= "main_sector", data = country_sector_wise_count_frame)
plt.xlabel("Country")
plt.ylabel("Count")
plt.show();


