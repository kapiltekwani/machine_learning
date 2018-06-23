import pandas as pd;

loan = pd.read_csv('loan.csv', low_memory=False)

print("Initialize size of load dataset", loan.shape)

########################################################################################################################

# First remove the customer behaviour related columns that does not make any sense

customer_behaviour_columns = [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "next_pymnt_d",
  "last_credit_pull_d",
  "application_type"
]

loan.drop(customer_behaviour_columns, axis=1, inplace=True)

print("Size of load dataset after removing customer behavior related columns", loan.shape)


########################################################################################################################

# Removing columns which have very less values or columns that dont add any value

meaning_less_variables = [
  "member_id","id",
  "acc_now_delinq",
  "chargeoff_within_12_mths",
  "pymnt_plan","initial_list_status",
  "delinq_amnt","pub_rec_bankruptcies",
  "tax_liens",
  "collections_12_mths_ex_med",
  "policy_code",
  "url","desc",
  "emp_title",
  "zip_code",
  "addr_state",
  "title"
]


loan.drop(meaning_less_variables, axis=1, inplace=True)

print("Size of load dataset after removing meaning_less_variables columns", loan.shape)


########################################################################################################################

# Dropping columns which have NA values for all columnes

loan.dropna(axis=1, how='all',inplace=True)

print("Size of load dataset after removing all columns which have just NA values", loan.shape)

########################################################################################################################

#Checking for all null values in the system
print(loan.isnull().sum())

print("deleting 'mths_since_last_delinq' column")
del loan['mths_since_last_delinq']

print("deleting 'mths_since_last_record' column")
del loan['mths_since_last_record']

print("Size of load dataset after removing all columns which have just NA values", loan.shape)

########################################################################################################################

loan.dropna(inplace=True)

current_loans = loan[loan.loan_status == "Current"]

loan = loan[loan.loan_status != "Current"]

loan['loan_status'] = loan["loan_status"].apply(lambda x: 1 if x == "Charged Off" else 0)

print(loan.shape)
print(loan.describe())

########################################################################################################################

## Finding Top 4 purpose for which people are taking Loan
temporary_df = loan.groupby('purpose')['loan_amnt'].count();
purpose_categorization_df = pd.DataFrame({'category': temporary_df.index, 'count': temporary_df.values}).sort_values(by='count', ascending=False)

# From purpose based categorization, we see that mostly
# loan is taken mostly for below mentioned 5 categories
#   debt_consolidation
#   credit_card   4899
#   other   3713
#   home_improvement   2785
#   major_purchase   2080
#

print(purpose_categorization_df)

## Going ahead for all analysis we will be using only these 4 purpose categories

loan = loan[loan.purpose.isin(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'])]

########################################################################################################################

## performing univariate analysis of year for all 4 popular purpose types

loan = loan[loan.purpose.isin(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'])]

loan['year'] =  "20" + loan['issue_d'].str.split("-").str[1]

year_purpose_categorization = loan.groupby(['year', 'purpose']).size()

print("performing univariate analysis of year for all 4 popular purpose types", year_purpose_categorization)

########################################################################################################################

## performing univariate analysis of term for all 4 popular purpose types

term_purpose_categorization = loan.groupby(['term', 'purpose']).size()

print("performing univariate analysis of term for all 4 popular purpose types", term_purpose_categorization)


########################################################################################################################

## performing univariate analysis of grade for all 4 popular purpose types

grade_purpose_categorization = loan.groupby(['grade', 'purpose']).size()

print("performing univariate analysis of grade for all 4 popular purpose types", grade_purpose_categorization)

########################################################################################################################

## performing univariate analysis of house_ownership for all 4 popular purpose types

house_purpose_categorization = loan.groupby(['home_ownership', 'purpose']).size()

print("performing univariate analysis of house_ownership for all 4 popular purpose types", house_purpose_categorization)


########################################################################################################################

## performing univariate analysis of verification_status for all 4 popular purpose types

house_purpose_categorization = loan.groupby(['verification_status', 'purpose']).size()

print("performing univariate analysis of verification_status for all 4 popular purpose types", house_purpose_categorization)

########################################################################################################################

loan['int_rate'] = loan['int_rate'].map(lambda x: x.rstrip('%')).astype('float64')

print("Mean Interest rate accross each grade", loan.groupby('grade')['int_rate'].mean())

print("Mean Interest rate accross each type of loan", loan.groupby('purpose')['int_rate'].mean())

# loan.to_csv('my_loans.csv')
########################################################################################################################

def create_loan_amount_range(x):
  if (x <=5000):
    return "small"
  elif (x > 5000 and x <= 15000):
    return "medium"
  elif (x > 15000 and x <= 25000):
    return "high"
  else :
    return "very_high"


def create_installment_range(x):
  if (x <=200):
    return "small"
  elif (x > 200 and x <= 400):
    return "medium"
  elif (x > 400 and x <= 600):
    return "high"
  else :
    return "very_high"

def create_income_range(x):
  if (x <=50000):
    return "small"
  elif (x > 50000 and x <= 100000):
    return "medium"
  elif (x > 100000 and x <= 150000):
    return "high"
  else :
    return "very_high"

def create_int_rate_range(x):
  if (x <= 10):
    return "low_rate"
  elif (x > 10 and x <= 15):
    return "medium_rate"
  else:
    return "high_rate"

def create_dti_range(x):
  if (x <= 10):
    return "low_dti"
  elif (x > 10 and x <= 20):
    return "medium_dti"
  else:
    return "high_dti"


def create_experience_range(x):
  if x == '< 1 year':
    return "fresher"
  elif x == '1 year' or x == '2 years' or x == '3 years':
    return "junior"
  elif x == '4 years' or x == '5 years' or x == '6 years'  or x == '7 years':
    return "senior"
  else:
    return "expert"

loan["range_loan_amnt"] = loan["loan_amnt"].map(lambda x: create_loan_amount_range(x))
loan["range_funded_amnt_inv"] = loan["funded_amnt_inv"].map(lambda x: create_loan_amount_range(x))
loan["range_int_rate"] = loan["int_rate"].map(lambda x: create_int_rate_range(x))
loan["range_dti"] = loan["dti"].map(lambda x: create_dti_range(x))
loan["range_funded_amnt"] = loan["funded_amnt"].map(lambda x: create_loan_amount_range(x))
loan["range_installment"] = loan["installment"].map(lambda x: create_installment_range(x))
loan["range_annual_inc"] = loan["annual_inc"].map(lambda x: create_income_range(x))
loan["emp_length"] = loan["emp_length"].map(lambda x: create_experience_range(x))



loan.to_csv('my_loans.csv')