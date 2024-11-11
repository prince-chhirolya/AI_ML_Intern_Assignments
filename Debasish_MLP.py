#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:


# Import College Data
college_data = pd.read_excel('C:/Users/Hp/Downloads/colleges.xlsx', header=0)


# In[3]:


# Melt the DataFrame to transform Tier columns into rows
college_data = pd.melt(college_data, var_name='Tier', value_name='College')


# In[4]:


# Remove NaN values
college_data = college_data.dropna()


# In[5]:


# Rename the columns
college_data = college_data.rename(columns={'College': 'College', 'Tier': 'Tier'})


# In[6]:


# Remove the 'Tier ' prefix from the Tier column
college_data['Tier'] = college_data['Tier'].str.replace('Tier ', '')


# In[8]:


# Now the data in the college dataset is reorganised with columns as College and Tier as below


# In[9]:


college_data.head()


# In[11]:


# Import the ML Case Stydy Data
ml_project = pd.read_excel('C:/Users/Hp/Downloads/ML case Study.xlsx')


# In[12]:


# Merge the college Data with ML Case Stydy Data on columns with respective Tier information
ml_project = pd.merge(ml_project, college_data[['College', 'Tier']], on='College')


# In[13]:


# Create Dummy Variables for Tier
Tier_dummies = pd.get_dummies(ml_project['Tier'], prefix='Tier')


# In[14]:


# Convert the dummy variables to 0 and 1 format
Tier_dummies = Tier_dummies.astype(int)


# In[15]:


# Now concat the Tier_dummies Data with ml_project
ml_project = pd.concat([ml_project, Tier_dummies], axis=1)


# In[16]:


# Now Drop the irrelevent columns 
ml_project = ml_project.drop(['College', 'Tier_3', 'Tier'], axis=1)


# In[17]:


# check the data
ml_project.head()


# In[18]:


# same for the cities Data set as College
# Load the cities dataset
cities_data = pd.read_excel('C:/Users/Hp/Downloads/cities.xlsx')


# In[19]:


# Melt the DataFrame to transform Metro/Non-Metro columns into rows
cities_data = pd.melt(cities_data, var_name='City_Type', value_name='City')


# In[20]:


# Remove NaN values
cities_data = cities_data.dropna()


# In[21]:


# check the data
cities_data.head()


# In[22]:


# Merge with ML Project
ml_project = pd.merge(ml_project, cities_data[['City', 'City_Type']], left_on='City', right_on='City')


# In[23]:


# Drop the city column as it is replaced by City_type
ml_project = ml_project.drop(['City'], axis=1)


# In[24]:


# Replace the Metro city with 1 and non-metro with 0
ml_project['City_Type'] = ml_project['City_Type'].replace(['Metrio City', 'non-metro cities'], [1, 0])


# In[25]:


# check the ML_project data
ml_project.head()


# In[26]:


# Now create Dummies to get rid of the categorical values in Role
ml_project = pd.get_dummies(ml_project)


# In[27]:


# Now drop the Role_Manager column as only n-1 columns required while creating dummies
ml_project = ml_project.drop(['Role_Manager'], axis=1)


# In[28]:


# Rename the City_Type column to Metro_city
ml_project = ml_project.rename(columns={'City_Type': 'Metro_city'})


# In[29]:


# change the boolean values of Role_Executive to 0 and 1 format
ml_project['Role_Executive'] = ml_project['Role_Executive'].astype(int)


# In[30]:


# Now check the data in ml project deprived of categorical values
ml_project.head()


# In[31]:


ml_project.info()


# In[32]:


# As all rows of the columns contain Data , No need for imputation


# In[33]:


# Now We will do Regression Technique. 
# 1 . MLR by using sklearn Library


# In[34]:


x_multi = ml_project.drop("CTC",axis = 1)


# In[35]:


Y_multi = ml_project['CTC']


# In[36]:


from sklearn.linear_model import LinearRegression
lm1 = LinearRegression()


# In[37]:


lm1.fit(x_multi,Y_multi)


# In[38]:


# test Train split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x_multi,Y_multi,test_size = 0.2,random_state = 0) 


# In[39]:


print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[40]:


lm_a = LinearRegression()


# In[41]:


lm_a.fit(X_train,Y_train)


# In[42]:


Y_test_a =lm_a.predict(X_test)


# In[43]:


Y_train_a = lm_a.predict(X_train)


# In[44]:


# check r2 score 


# In[45]:


from sklearn.metrics import r2_score


# In[46]:


r2_score(Y_test,Y_test_a)


# In[47]:


r2_score(Y_train,Y_train_a)


# In[48]:


# Hence the r2 score by MLR is 0.5888692548800096


# In[49]:


# Shrinkage method
# 1. Ridge regression


# In[50]:


# First Preprocess the Data 


# In[51]:


from sklearn import preprocessing


# In[52]:


scaler = preprocessing.StandardScaler().fit(X_train)


# In[53]:


X_train_s =scaler.transform(X_train)


# In[54]:


X_test_s = scaler.transform(X_test)


# In[55]:


from sklearn.linear_model import Ridge


# In[56]:


lm_r = Ridge(alpha = 0.5) 


# In[57]:


lm_r.fit(X_train_s,Y_train)


# In[59]:


r2_score(Y_test,lm_r.predict(X_test_s))


# In[60]:


# Hence r2 score in Ridge regression is 0.5888248509763372


# In[61]:


# Now with multiple values of alpha
from sklearn.model_selection import validation_curve


# In[62]:


param_range = np.logspace(-2,8,100)


# In[65]:


train_scores, test_scores = validation_curve(
    Ridge(), 
    X_train_s, 
    Y_train, 
    param_name="alpha", 
    param_range=param_range, 
    scoring='r2'
)


# In[66]:


print(train_scores)


# In[67]:


print(test_scores)


# In[68]:


train_mean = np.mean(train_scores,axis=1)


# In[69]:


test_mean = np.mean(test_scores,axis=1)


# In[70]:


max(test_mean)


# In[71]:


np.where(test_mean==max(test_mean))


# In[72]:


param_range[33]


# In[73]:


lm_r_best = Ridge(alpha= param_range[33])


# In[75]:


lm_r_best.fit(X_train_s,Y_train)


# In[76]:


r2_score(Y_test,lm_r_best.predict(X_test_s))


# In[77]:


r2_score(Y_train,lm_r_best.predict(X_train_s))


# In[78]:


# Hence after using the multiple values of alpha and picking the best one the r2 value is 0.5868537675116259


# In[79]:


# 3.Lasso


# In[80]:


from sklearn.linear_model import Lasso


# In[81]:


lm_l = Lasso(alpha = 0.4)


# In[82]:


lm_l.fit(X_train_s,Y_train)


# In[83]:


r2_score(Y_test,lm_r.predict(X_test_s))	


# In[84]:


train_scores_l, test_scores_l = validation_curve(
    Lasso(), 
    X_train_s, 
    Y_train, 
    param_name="alpha", 
    param_range=param_range, 
    scoring='r2'
)


# In[85]:


train_mean_l = np.mean(train_scores_l,axis=1)


# In[86]:


test_mean_l = np.mean(test_scores_l,axis=1)


# In[87]:


max(test_mean_l)


# In[88]:


np.where(test_mean_l==max(test_mean_l))


# In[89]:


param_range[37]


# In[90]:


lm_l_best = Lasso(alpha= param_range[37])


# In[91]:


lm_l_best.fit(X_train_s,Y_train)


# In[92]:


r2_score(Y_test,lm_l_best.predict(X_test_s))


# In[93]:


r2_score(Y_train,lm_l_best.predict(X_train_s))


# In[94]:


# hence the r2 score using lasso is 0.5872179885197018


# In[97]:


# 4. Now we will Run KNN


# In[96]:


# First We will standardise the data


# In[98]:


from sklearn import preprocessing


# In[99]:


scaler = preprocessing.StandardScaler().fit(X_train)


# In[100]:


X_train_s = scaler.transform(X_train)


# In[101]:


scaler = preprocessing.StandardScaler().fit(X_test)


# In[102]:


X_test_s = scaler.transform(X_test)


# In[103]:


from sklearn.neighbors import KNeighborsRegressor


# In[104]:


clf_knn_1 = KNeighborsRegressor(n_neighbors=1)


# In[105]:


clf_knn_1.fit(X_train_s,Y_train)


# In[116]:


# for multiple values of n_neighbors_values
n_neighbors_values = [1, 5, 10, 20, 50]


# In[109]:


best_n_neighbors = None


# In[110]:


best_r2_score = -float('inf')  # Initialize with a very low value


# In[117]:


for n in n_neighbors_values:
    clf_knn = KNeighborsRegressor(n_neighbors=n)
    clf_knn.fit(X_train, Y_train)
    r2_score = clf_knn.score(X_test, Y_test)  # Evaluate on test data
    print(f"R2 score for n_neighbors={n}: {r2_score}")


# In[118]:


# Hence r2 value with KNN analysis with n_neighbors=10 = 0.10503749427390807


# In[119]:


#5. Decision Tree - Bagging :


# In[122]:


clftree = DecisionTreeRegressor()


# In[123]:


from sklearn.ensemble import BaggingRegressor


# In[124]:


bag_clf = BaggingRegressor(base_estimator=clftree, n_estimators=1000,
bootstrap=True,n_jobs=-1,random_state=42)


# In[125]:


bag_clf.fit(X_train, Y_train)


# In[126]:


# Test R2 score
test_r2_score = bag_clf.score(X_test, Y_test)
print("Test R2 score: ", test_r2_score)


# In[127]:


# Hence r2 value with Bagging is 0.710323969897245


# In[128]:


# 6. Random Forest


# In[129]:


from sklearn.ensemble import RandomForestRegressor


# In[130]:


rf_clf = RandomForestRegressor(n_estimators = 1000, n_jobs=-1,random_state=42)


# In[131]:


rf_clf.fit(X_train, Y_train)


# In[132]:


# Test R2 score
test_r2_score = rf_clf.score(X_test, Y_test)
print("Test R2 score: ", test_r2_score)


# In[133]:


# Hence r2 value with RF is 0.7109592255042925


# In[134]:


# using GridsearchCV


# In[135]:


from sklearn.model_selection import GridSearchCV


# In[136]:


rf_clf = RandomForestRegressor(random_state = 42)


# In[137]:


params_grid = {"max_features": [4,5,6,7,8,9,10],"min_samples_split": [2,3,10]}


# In[138]:


grid_search = GridSearchCV(rf_clf, params_grid ,n_jobs=-1, scoring='accuracy')


# In[139]:


grid_search.fit(X_train, Y_train)


# In[140]:


grid_search.best_params_


# In[141]:


cvrf_clf = grid_search.best_estimator_


# In[142]:


# Test R2 score
test_r2_score = cvrf_clf.score(X_test, Y_test)
print("Test R2 score: ", test_r2_score)


# In[3]:


# Hence r2 value with Grid values is 0.6980440448697041


# In[4]:


pip install nbconvert


# In[5]:


nbconvert --to python Debasish_MLP.ipynb


# In[1]:


nbconvert --to python Debasish_MLP.ipynb


# In[2]:


pip install nbconvert


# In[ ]:




