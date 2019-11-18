
# coding: utf-8

# In[1]:


# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#from pandas import compat
from IPython.core.display import HTML
sns.set_style('whitegrid')

#compat.PY3 = False


# In[2]:


#Importing dataset
carprice = pd.read_csv('CarPrice_Assignment.csv',sep=",",encoding="utf-8",low_memory=False)
#media = media.drop('Unnamed: 7',axis = 1)
#carprice.columns = [col.decode('ascii', 'ignore') for col in carprice]

carprice.head(1)


# In[3]:


#Change lowercase to all column headers
carprice.columns = map(str.lower, carprice.columns)


# In[4]:


carprice.info()


# In[5]:


carprice['symboling']=carprice['symboling'].astype('object')


# In[6]:


carprice.columns


# In[7]:


carprice.head()


# In[8]:


carprice.drivewheel.describe()


# In[9]:


carprice.isnull().sum()


# In[10]:


carprice.shape


# In[11]:


carprice['fueltype'].value_counts()


# In[12]:


carprice['aspiration'].value_counts()


# In[13]:


carprice['carconame']=carprice['carname'].str.split().str[0]


# In[14]:


carprice.head(1)


# In[15]:


carprice.head(1)


# In[16]:


carprice['carbody'].unique()
carprice['carconame'] = carprice['carconame'].str.lower()


# In[17]:


#carprice[columns_dtype_object].apply(lambda col: print(col.unique()), axis=0)
carprice[['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem','carconame']].apply(lambda col: print(col.unique()),axis=0)


# In[18]:


#carprice['carconame'] = np.where(carprice['carconame'] == 'vw', 'vokswagen','vw')
carprice.loc[carprice['carconame'] == 'vw', 'carconame'] = 'volkswagen'
carprice.loc[carprice['carconame'] == 'vokswagen', 'carconame'] = 'volkswagen'
carprice.loc[carprice['carconame'] == 'maxda', 'carconame'] = 'mazda'


carprice.loc[carprice['carconame'] == 'toyouta', 'carconame'] = 'toyota'
carprice.loc[carprice['carconame'] == 'porcshce', 'carconame'] = 'porsche'
carprice.loc[carprice['drivewheel'] == '4wd', 'drivewheel'] = 'fwd'


# In[19]:


carprice['enginetype'].unique()
carprice.loc[carprice['doornumber'] == 'two', 'doornumber'] = 'two_door'
carprice.loc[carprice['doornumber'] == 'four', 'doornumber'] = 'four_door'
carprice.loc[carprice['cylindernumber'] == 'four', 'cylindernumber'] = 'four_cyl'
carprice.loc[carprice['cylindernumber'] == 'six', 'cylindernumber'] = 'six_cyl'
carprice.loc[carprice['cylindernumber'] == 'five', 'cylindernumber'] = 'five_cyl'
carprice.loc[carprice['cylindernumber'] == 'three', 'cylindernumber'] = 'three_cyl'
carprice.loc[carprice['cylindernumber'] == 'twelve', 'cylindernumber'] = 'twelve_cyl'
carprice.loc[carprice['cylindernumber'] == 'two', 'cylindernumber'] = 'two_cyl'
carprice.loc[carprice['cylindernumber'] == 'eight', 'cylindernumber'] = 'eight_cyl'


# In[20]:


carprice['carconame'].unique()


# In[21]:


# Converting STRING TO Categorical Variables
#carprice['cylindernumber1'] = carprice['cylindernumber'].map({u'four':3, u'six':5, u'five':4, u'three':2, u'twelve':7, u'two':1, u'eight':6})
carprice['fueltype'] = carprice['fueltype'].map({'gas': 1, 'diesel': 0})
carprice['aspiration'] = carprice['aspiration'].map({'std': 1, 'turbo': 0})
carprice['doornumber'] = carprice['doornumber'].map({'two_door': 1, 'four_door': 0})

carprice['drivewheel'] = carprice['drivewheel'].map({'fwd': 1, 'rwd': 0})
carprice['enginelocation'] = carprice['enginelocation'].map({'front': 1, 'rear': 0})

carbody = pd.get_dummies(carprice['carbody'],drop_first=True)
cylindernumber = pd.get_dummies(carprice['cylindernumber'],drop_first=True)
fuelsystem = pd.get_dummies(carprice['fuelsystem'],drop_first=True)
enginetype = pd.get_dummies(carprice['enginetype'],drop_first=True)
carconame = pd.get_dummies(carprice['carconame'],drop_first=True)

carconame.shape


# In[22]:


carprice.shape


# In[23]:


carprice.head()


# In[24]:


carprice = pd.concat([carprice,carbody,fuelsystem,enginetype,cylindernumber,carconame],axis=1)
carprice.shape


# In[25]:


carprice.head()


# In[26]:


print(carprice.columns.tolist())


# In[27]:


# Converting Yes to 1 and No to 0

carprice.drop(['car_id','carname','carbody','cylindernumber','fuelsystem','enginetype','carconame'],axis=1,inplace=True)
carprice.shape


# In[28]:


carprice.head()


# In[29]:


carprice.describe()


# In[30]:


### Rescaling the Features 
#defining a normalisation function 
def normalize (x): 
    return ( (x-np.mean(x))/ (max(x) - min(x)))
                                            
                                              
# applying normalize ( ) to all columns 
carprice = carprice.apply(normalize)


# In[31]:


carprice.shape


# In[32]:


carprice.head()


# In[33]:


carprice.shape


# In[34]:


carprice.columns
#64 columns


# In[35]:


# Putting feature variable to X
data_final_vars=carprice.columns.values.tolist()
y_cols=['price']
X_cols=[i for i in data_final_vars if i not in y_cols]
X= carprice[X_cols]
y= carprice[y_cols]

print(type(X))
print(type(y))

print(X.shape)
print(y.shape)
print(X.head(1))
print(y.head(1))


# In[36]:


print(X.shape)
print(y.shape)
#y = y[:,np.newaxis]
print(y.shape)


# In[37]:


# Putting feature variable to X
#X = carprice[['symboling', 'fueltype', 'aspiration', 'doornumber', 'drivewheel','enginelocation', 'wheelbase', 'carlength', 'carwidth', 'carheight','curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio','horsepower', 'peakrpm', 'citympg', 'highwaympg', 'hardtop',       'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi','spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'rotor','five_cyl', 'four_cyl', 'six_cyl', 'three_cyl', 'twelve_cyl', 'two_cyl','audi', 'bmw', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu','jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan','peugeot', 'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota','volkswagen', 'volvo']]

# Putting response variable to y
#y = carprice['price']


# In[38]:


print(type(X))

print(type(y))


# In[39]:


print(X.shape)
print(y.shape)
#y = y[:,np.newaxis]
print(y.shape)


# In[40]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)
print(X_train.shape)
print(y_train.shape)
#y_train = y_train[:,np.newaxis]
print(y_train.shape)


# In[41]:


# UDF for calculating vif value
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)


# In[42]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[43]:


# Running RFE with the output number of the variable equal to 9
lm = LinearRegression()
rfe = RFE(lm, 15)             # running RFE
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)  


# In[44]:


col = X_train.columns[rfe.support_]
print(col)


# In[45]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[46]:


import statsmodels.api as sm  


# In[47]:


# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)


# In[48]:


lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
print(X_train_rfe.shape)
print(y_train.shape)

print(y_test.shape)
#y_test = y_test[:,np.newaxis]
print(y_test.shape)


# In[49]:


#Let's see the summary of our linear model
print(lm.summary())


# In[50]:


#y_pred=model.predict(sm.tools.tools.add_constant(X_test,prepend=True,has_constant='add'))
##y_pred=lm.predict(sm.tools.tools.add_constant(X_test,prepend=True,has_constant='add'))


# In[51]:


col = X_train.columns[rfe.support_]
print(col)


# In[52]:


vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo','aspiration','drivewheel','horsepower','fueltype','doornumber'],axis=1),dependent_col="price")


# In[53]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


plt.figure(figsize = (15,19))
sns.heatmap(X_train_rfe.corr(),cmap="YlGnBu",annot = True)


# In[55]:


## Making Predictions
# Now let's use our model to make predictions.

# Creating X_test_6 dataframe by dropping variables from X_test
#X_test_rfe = X_test[col]

# Adding a constant variable 
#X_test_rfe = sm.add_constant(X_test_rfe,has_constant='add')

# Making predictions
#y_pred = lm.predict(X_test_rfe)


# In[56]:


#X_test_rfe.shape


# In[58]:


# Actual and Predicted (this model is not predictind well as unable to answer the peaks)
import matplotlib.pyplot as plt
#c = [i for i in range(1,63,1)] # generating index 
#fig = plt.figure() 
#plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
#plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
#fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
#plt.xlabel('Index', fontsize=18)                               # X-label
#plt.ylabel('Housing Price', fontsize=16)                       # Y-label


# In[59]:


#print(y_test.shape)
#print(type(y_test))

#print(type(y_pred))
#print(y_pred.shape)


# In[60]:


#df = pd.DataFrame(university_guide.values.reshape((30, 2)), columns=['Rank', 'University'])
#y_pred= pd.DataFrame(y_pred.values.reshape((62,1)))
#print(type(y_pred))
#print(y_pred.shape)
#print(y_test.shape)


# In[65]:


X_train_rfe.columns


# In[66]:


### Iteration No-2
# New X_var after dropping rotor with VIF as inf
# Dropping highly correlated variables and insignificant variables
X_train_rfe = X_train_rfe.drop('rotor', 1)


# In[67]:


# Create a second fitted model
lm_2 = sm.OLS(y_train,X_train_rfe).fit()


# In[68]:


#Let's see the summary of our second linear model
print(lm_2.summary())


# In[69]:


# Calculating Vif value
#vif_cal(input_data=housing.drop(["bbratio"], axis=1), dependent_col="price")
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor'],axis=1),dependent_col="price")


# In[70]:


# 32rd iteration -Dropping next highly correlated variable enginesize and insignificant variables
X_train_rfe = X_train_rfe.drop('enginesize', 1)


# In[71]:


# Create a third fitted model
lm_3 = sm.OLS(y_train,X_train_rfe).fit()


# In[72]:


#Let's see the summary of our second linear model
print(lm_3.summary())


# In[73]:


# Calculating Vif value
#VIF after removing enginesize
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize'],axis=1),dependent_col="price")


# In[74]:


plt.figure(figsize = (15,19))
sns.heatmap(X_train_rfe.corr(),cmap="YlGnBu",annot = True)


# In[75]:


# Dropping next highly correlated variable curb weight with carwidth but p values are 0 so remove next high VIF variable four_cyl with highest  p value
X_train_rfe = X_train_rfe.drop('four_cyl', 1)


# In[76]:


X_train_rfe.columns


# In[77]:


# Create a 4th fitted model
lm_4 = sm.OLS(y_train,X_train_rfe).fit()


# In[78]:


#Let's see the summary of our second linear model
print(lm_4.summary())


# In[79]:


# Calculating Vif value
#VIF after removing 
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl'],axis=1),dependent_col="price")


# In[80]:


# Dropping highly correlated variables curb weight
X_train_rfe = X_train_rfe.drop('curbweight', 1)
# Create a second fitted model
lm_5 = sm.OLS(y_train,X_train_rfe).fit()
#Let's see the summary of our second linear model
print(lm_5.summary())


# In[81]:


# Calculating Vif value
#VIF after removing 
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl','curbweight'],axis=1),dependent_col="price")


# In[82]:


# Dropping highly correlated variables and insignificant variables
X_train_rfe = X_train_rfe.drop('five_cyl', 1)
# Create a second fitted model
lm_6 = sm.OLS(y_train,X_train_rfe).fit()
#Let's see the summary of our second linear model
print(lm_6.summary())


# In[83]:


# Calculating Vif value
#VIF after removing five_cyl
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl','curbweight','five_cyl'],axis=1),dependent_col="price")


# In[84]:


# Dropping highly correlated variables and insignificant variables
X_train_rfe = X_train_rfe.drop('porsche', 1)
# Create a second fitted model
lm_7 = sm.OLS(y_train,X_train_rfe).fit()
#Let's see the summary of our second linear model
print(lm_7.summary())


# In[85]:


# Calculating Vif value
#VIF after removing porsche
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen','volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl','curbweight','five_cyl','porsche'],axis=1),dependent_col="price")


# In[86]:


# Dropping highly correlated variables and insignificant variables
X_train_rfe = X_train_rfe.drop('two_cyl', 1)
# Create a second fitted model
lm_8 = sm.OLS(y_train,X_train_rfe).fit()
#Let's see the summary of our second linear model
print(lm_8.summary())


# In[87]:


# Calculating Vif value
#VIF after removing two_cyl
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen','volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl','curbweight','five_cyl','porsche','two_cyl'],axis=1),dependent_col="price")


# In[88]:


# Dropping highly correlated variables and insignificant variables
X_train_rfe = X_train_rfe.drop('stroke', 1)
# Create a second fitted model
lm_9 = sm.OLS(y_train,X_train_rfe).fit()
#Let's see the summary of our second linear model
print(lm_9.summary())


# In[89]:


# Calculating Vif value
#VIF after removing stroke
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen','volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl','curbweight','five_cyl','porsche','two_cyl','stroke'],axis=1),dependent_col="price")


# In[90]:


# Dropping highly correlated variables and insignificant variables
X_train_rfe = X_train_rfe.drop('boreratio', 1)
# Create a second fitted model
lm_10 = sm.OLS(y_train,X_train_rfe).fit()
#Let's see the summary of our second linear model
print(lm_10.summary())


# In[91]:


# Calculating Vif value
#VIF after removing boreratio
vif_cal(input_data=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen','volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl','curbweight','five_cyl','porsche','two_cyl','stroke','boreratio'],axis=1),dependent_col="price")


# In[ ]:


# Now let's check how well our model is able to make predictions.

# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[112]:


new_coldf=carprice.drop(['symboling', 'wheelbase', 'carlength', 'carheight', 'compressionratio', 'peakrpm', 'citympg', 'highwaympg', 'hardtop', 'hatchback', 'sedan', 'wagon', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi', 'dohcv', 'l', 'ohc', 'ohcf', 'ohcv', 'six_cyl', 'audi', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'plymouth', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen','volvo','aspiration','drivewheel','horsepower','fueltype','doornumber','rotor','enginesize','four_cyl','curbweight','five_cyl','porsche','two_cyl','stroke','boreratio','price'],axis=1)


# In[113]:


new_col=new_coldf.columns
print(new_col.shape)
#new_col=pd.DataFrame(new_col.values.reshape((7,1)))
print(new_col.shape)


# In[116]:


## Making Predictions
# Now let's use our model to make predictions.

# Creating X_test_6 dataframe by dropping variables from X_test
X_test_rfe = X_test[new_col]

# Adding a constant variable 
X_test_rfe = sm.add_constant(X_test_rfe,has_constant='add')

# Making predictions
y_pred_lm_10 = lm_10.predict(X_test_rfe)


# In[117]:


print(y_pred.shape)
print(y_pred.shape)


# In[118]:


# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Actual and Predicted (this model is not predictind well as unable to answer the peaks)
import matplotlib.pyplot as plt
c = [i for i in range(1,63,1)] # generating index 
fig = plt.figure() 
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
plt.plot(c,y_pred_lm_10, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)                       # Y-label


# In[123]:



error= pd.DataFrame(y_test-y_pred_lm_10.values.reshape((62,1)))
print(error.shape)
print(y_test.shape)
print(y_pred_lm_10.shape)
display(error.head(1))
display(y_test.head(1))
display(y_pred_lm_10.head(1))


# In[124]:



# Error terms
c = [i for i in range(1,63,1)]
fig = plt.figure()

plt.plot(c,y_test-y_pred_lm_10, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,error, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred_lm_10', fontsize=16)                # Y-label


# In[125]:



# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred_lm_10)
fig.suptitle('y_test vs y_pred_lm_10', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label



# In[126]:


# Now let's check the Root Mean Square Error of our model.
import numpy as np
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lm_10)))

