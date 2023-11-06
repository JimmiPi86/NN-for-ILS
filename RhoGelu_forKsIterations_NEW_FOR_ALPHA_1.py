#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import basic libraries
import numpy as np
import pandas as pd
import seaborn as sns

# Import r2_score functions from scikit-learn library to evaluate regression models and MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Import cross_val_score and train_test_split functions from scikit-learn library
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Import the MinMaxScaler class from the scikit-learn library for scaling data
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Load data from CSV file 'merged_data1.csv' using 'utf8' encoding and ';' separator
df = pd.read_csv('merged_data1.csv',encoding = 'utf8',sep=';')
df.head() #Shows the first 5 rows of the data frame


# In[3]:


'''Select the columns 'M_C', 'M_A', 'IS_SYM', 'P', 'T' and store them as a NumPy matrix in the variable 'X' 
Where:
M_C : Molar mass of cation
M_A: Molar mass of anion
IS_SYM: Is the cation symmetrical (0,1)
P: Pressure (MPa)
T: Temperature (K)

'''
X = df[['M_C', 'M_A', 'IS_SYM', 'P', 'T']].values
#Select the 'Rho' column and store it as a NumPy matrix in the 'y' variable (Rho - density kg/m^3)
y = df[['Rho']].values


# In[4]:


'''Data preparation - dividing data into training and test set'''

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size = 0.3, random_state = 42)

print(X_train.shape)
print(y_train.shape)
print('Test shapes')
print(X_test.shape)
print(y_test.shape)


# In[5]:


'''Initialize the MinMaxScaler object and Fit scaling to training data 'X_train' 
and Scale training data 'X_train' and test data 'X_test' '''
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # Main architecture of the network

# In[6]:


# Import the Sequential class from the TensorFlow/Keras library to create sequential models
from tensorflow.keras.models import Sequential

# Import a Dense layer from the TensorFlow/Keras library to create dense layers.
from tensorflow.keras.layers import Dense

# Import a Dropout layer from the TensorFlow/Keras library to apply discard layers.
from keras.layers import Dense, Dropout

# Import l2 regularization from Keras library to L2 regularization
from keras.regularizers import l2


# ## NN Model

# In[7]:


'''
The code creates a sequential deep neural network 
model with several densely connected layers. 
The model has different numbers of neurons in successive layers, 
different activation functions (tanh and GELU) and L2 regularization to control overfitting. 
The last layer with one neuron suggests that the model is used for a regression task.
'''

model = Sequential()
model.add(Dense(5, activation='tanh', kernel_regularizer=l2(0.1)))  # Model with strong L2 reg.
model.add(Dense(5, activation='tanh', kernel_regularizer=l2(0.1)))
model.add(Dense(55, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(55, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(25, activation='gelu', kernel_regularizer=l2(0.1)))
model.add(Dense(1))


# In[8]:


n = 1800


# In[9]:


#Compile and train network on n epochs

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])
history = model.fit(x=X_train, y=y_train, epochs=n, verbose=0)


# In[10]:


# Predict values on the test set, train set and store the results in the variable 'y_test_m & y_train_m'

y_test_m=model.predict(X_test)
y_train_m= model.predict(X_train)


# In[11]:


### Metrics for test set:
r2 = r2_score(y_test_m, y_test)
print(f"Determination coefficient R^2: {r2}")
mse = mean_squared_error(y_test_m, y_test)
print(f"Mean squared error MSE: {mse}")


# In[12]:


### Metrics for train set:
r2_train = r2_score(y_train_m, y_train)
print(f"Determination coefficient R^2: {r2_train}")
mse_train = mean_squared_error(y_train, y_train)
print(f"Mean squared error MSE: {mse_train}")


# In[13]:


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
import keras
from scikeras.wrappers import KerasRegressor

from sklearn.model_selection import cross_val_score


# In[14]:


def create_model():
    return model

keras_regressor = KerasRegressor(build_fn=create_model, epochs=n, batch_size= 64)  # Ustaw odpowiednie parametry


# In[16]:


# Use cross-validation (cross_val_score) to evaluate the 'keras_regressor' model
# on the test set 'X_test' with five folds and evaluate the results using the R-squared ('r2') measure.

cv_scores = cross_val_score(keras_regressor, X_test, y_test, cv=5, scoring='r2')


# In[17]:


mean_r2 = cv_scores.mean()
print("Average R^2 after 5-k fold cross.val.:", mean_r2)


# In[18]:


cv_scores


# In[19]:


print(history.history.keys())


# In[20]:


model.evaluate(X_test,y_test)


# In[21]:


test_predictions = model.predict(X_test)


# In[23]:


train_predictions = model.predict(X_train)


# In[36]:


test_predictions


# In[37]:


# Convert the 'test_predictions' and train_predictions array to a Pandas series object so that it has the correct shape
test_predictions = pd.Series(test_predictions)


# In[34]:


train_predictions1 = pd.Series(train_predictions.reshape(2230,))


# In[40]:


pred_df = pd.DataFrame(y_test,columns = ['Test TRUE Y'])


# In[41]:


pred_df = pd.concat([pred_df,test_predictions],axis = 1)


# In[42]:


pred_df.columns = ['Test true y', 'Pred']


# In[43]:


train_df = pd.DataFrame(y_train,columns = ['Test TRUE Y'])


# In[44]:


train_df = pd.concat([train_df,train_predictions1],axis = 1)


# In[45]:


train_df.columns = ['Test true y', 'Pred']


# In[46]:


sns.scatterplot(x = 'Test true y', y = 'Pred', data = train_df)
sns.scatterplot(x = 'Test true y', y = 'Pred', data = pred_df, alpha = 0.2)


# In[47]:


train_df.head()


# In[48]:


pred_df.head()


# In[44]:


#Save full train and test set to CSV
train_df.to_csv('train_set_GESTOSC_GELU_ALPHA.csv', sep=';', encoding='utf-8')
pred_df.to_csv('test_set_GESTOSC_GELU_ALPHA.csv', sep=';', encoding='utf-8')


# In[45]:


#Save model for future

model.save("Model_Rho_ALPHA_G1.h5")


# # Check network predictions for specific ionic liquids

# In[62]:


def predictions3(MC, MA, SYM, P, T,vb=0):
    '''
    This function iterates over the pressure and temperature of the set ionic liquid.

    Parameters:
    MC (float): Mass of the cation.
    MA (float): Mass of the anion.
    SYM (int): Some constant value for IS_SYM.
    P (list): List of pressure values.
    T (list): List of temperature values.
    vb (int, optional): Verbosity level for model prediction. Defaults to 0.
    
    Returns:
    list: List of model predictions for all combinations of pressure and temperature.'''

    res = [model.predict(scaler.transform([[MC, MA, SYM, i, j]]), verbose=vb)[0] for i in P for j in T]
    return res


# In[75]:


# '''TEST FOR SPECIFIC ILS '''
name = 'C3C1Pyr_NTF2'
Mcat = 113.121
Man = 280.146

P = [0.1,10,20,30,40,50,60,70,80,90,100]
T = [293.15,298.15,303.15,308.15,313.15,318.15]
result = predictions4(Mcat,Man,0,P,T);


# In[64]:


res_flat = np.array(result).flatten()  # transformation to a one-dimensional numpy array
res_numerical = [val.item() for val in res_flat]  # extraction of numerical values


# In[69]:


#This is just to check "In cito"
elements = [element for element in res_numerical]
print(elements)


# In[71]:


#This part has the task of converting 1D arrays into an array of dimension according to the experimental data
data_table = np.array(res_numerical).reshape(len(P), len(T))
pressure = pd.DataFrame(P, columns=['P'])
new_headers = {'P': 'P'}
new_cols = T
for i, temperature in enumerate(new_cols):
    new_headers[i] = temperature
data = pd.concat([pressure, pd.DataFrame(data_table)], axis=1)
data.rename(columns=new_headers, inplace=True)


# In[72]:


#Check
data


# In[225]:


#Save dataframe to excel 
data.to_excel(name+'_PURE_R_DATA.xlsx', index=False)  


# # NN Model ends here

# # Upload model

# In[ ]:





# In[77]:


from tensorflow.keras.models import load_model
model = load_model("Model_Rho_ALPHA_G1.h5") #Model name 


# In[79]:


train_predictions = model.predict(X_train)


# In[81]:


def predictions3(MC, MA, SYM, P, T,vb=0):
    '''
    This function iterates over the pressure and temperature of the set ionic liquid.

    Parameters:
    MC (float): Mass of the cation.
    MA (float): Mass of the anion.
    SYM (int): Some constant value for IS_SYM.
    P (list): List of pressure values.
    T (list): List of temperature values.
    vb (int, optional): Verbosity level for model prediction. Defaults to 0.
    
    Returns:
    list: List of model predictions for all combinations of pressure and temperature.'''

    res = [model.predict(scaler.transform([[MC, MA, SYM, i, j]]), verbose=vb)[0] for i in P for j in T]
    return res


# In[82]:


name = 'C2ImC1OC6_NTF2'
Mcat = 211.181
Man = 280.146

P = [0.1019,9.81,19.62,29.43,39.24,49.05,58.86,68.67,78.48,88.29,98.1,107.91,117.72,127.53,137.34,147.15,156.96,166.77,176.58,186.39,196.2]
T = [293.75,312.85,333.15,352.95,373.25]
result = predictions3(Mcat,Man,0,P,T);


# In[83]:


res_flat = np.array(result).flatten()  # transformation to a one-dimensional numpy array
res_numerical = [val.item() for val in res_flat]  # extraction of numerical values


# In[84]:


#This part has the task of converting 1D arrays into an array of dimension according to the experimental data
data_table = np.array(res_numerical).reshape(len(P), len(T))
pressure = pd.DataFrame(P, columns=['P'])
new_headers = {'P': 'P'}
new_cols = T
for i, temperature in enumerate(new_cols):
    new_headers[i] = temperature
data = pd.concat([pressure, pd.DataFrame(data_table)], axis=1)
data.rename(columns=new_headers, inplace=True)


# In[86]:


data


# In[ ]:





# In[42]:


data.to_excel(name+'_PURE_R_DATA_AT.xlsx', index=False)  


# # Special cases for testing numerical differentiation. Theory: the more steps, the better the derivatives should come out.

# # Continuum
# 

# In[87]:


# Create a new list with increments of 1
new_P = list(range(int(P[0]), int(P[-1]) + 1))
new_P[0] = 0.1
# Count step
step = 1.0

# Tworzenie nowej listy z krokami co 1
new_T = [T[0]]
while new_T[-1] < T[-1]:
    new_T.append(new_T[-1] + step)


# ## P original, but T -> step 1K

# In[88]:


len(P)*len(new_T)


# In[90]:


import time
start_time = time.time()
result =predictions3(Mcat,Man,0,P,new_T)
end_time = time.time()
exec_time = end_time - start_time
print(f"Time for execution: {exec_time} sec")


# In[91]:


res_flat = np.array(result).flatten()  # przekształcenie do jednowymiarowej tablicy numpy
res_numerical = [val.item() for val in res_flat]  # wyodrębnienie wartości liczbowych
data_continuum = np.array(res_numerical).reshape(len(P), len(new_T))
cont = pd.DataFrame(data_continuum)
df = pd.DataFrame(P, columns=['P'])
data_continuum = pd.concat([df, cont], axis=1)


# In[94]:


new_headers = {'P': 'P'}
new_columns = new_T
for i, temperatures in enumerate(new_columns):
    new_headers[i] = temperatures
data_continuum.rename(columns=nowe_naglowki, inplace=True)    


# In[95]:


data_continuum


# In[30]:


data_continuum.to_excel(nazwa+'_ORIG_P_Step1T_R_DATA.xlsx', index=False)  


# ## p co 1 t co 1

# In[96]:


len(new_P)*len(new_T)


# In[97]:


import time
start_time = time.time()
result_1_1 =predictions3(Mcat,Man,0,new_P,new_T)
end_time = time.time()
execut_time = end_time - start_time
print(f"Time {execut_time} sec")


# In[98]:


res_flat_1_1 = np.array(result_1_1).flatten()  
res_numerical_1_1 = [val.item() for val in res_flat_1_1]  # wyodrębnienie wartości liczbowych
continuum_1_1 = np.array(res_numerical_1_1).reshape(len(new_P), len(new_T))
cont11 = pd.DataFrame(continuum_1_1)
df_11 = pd.DataFrame(new_P, columns=['P'])
continuum_1_1 = pd.concat([df_11, cont11], axis=1)
new_headers = {'P': 'P'}
new_cols = new_T
for i, temperatures in enumerate(new_cols):
    new_headers[i] = temperatures
cont11.rename(columns=new_headers, inplace=True) 


# In[101]:


cont11


# In[148]:


cont11.to_excel(nazwa+'_ORIG_1-1CONTINUUM_DATA.xlsx', index=False)  

