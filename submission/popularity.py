import numpy as np;
import pandas as pd;

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
pd.options.display.float_format = '{:.0f}'.format

print("******* Checkpoint #1 *****")

new_df = pd.read_csv('/Users/tkapil/Downloads/grades.csv')

new_df['Year'] = pd.DatetimeIndex(new_df['submit_time']).year
new_df['Month'] = pd.DatetimeIndex(new_df['submit_time']).month
new_df['Date'] = pd.DatetimeIndex(new_df['submit_time']).date
new_df['Hour'] = pd.DatetimeIndex(new_df['submit_time']).hour
new_df['Minute'] = pd.DatetimeIndex(new_df['submit_time']).minute
new_df['Second'] = pd.DatetimeIndex(new_df['submit_time']).second


submission_df = new_df[new_df['submit_time'] > "01/03/17-23:59:52:00" & new_df['submit_time'] < "01/09/17-23:59:52:00"]

print(submission_df.head())
# round2_df['permalink'] = round2_df.company_permalink.str.upper()
