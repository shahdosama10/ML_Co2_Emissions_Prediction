# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



# %% [markdown]
# ## a) Load the "co2_emissions_data.csv" dataset.

# %%
df = pd.read_csv('co2_emissions_data.csv')


# %% [markdown]
# 
# ## b) Perform analysis on the dataset to

# %% [markdown]
# 
# >#### i) check whether there are missing values

# %%

# to know the number of the rows
print(len(df))

# to get the number of non-null in each column
df.info()

# according to the output there aren't any missing values


# %% [markdown]
# >#### ii) check whether numeric features have the same scale

# %%
# To get min and max values for each column

df.describe().T

# according to the output the numeric features dosn't have the same scale


# %% [markdown]
# >#### iii) visualize a pairplot in which diagonal subplots are histograms

# %%
sns.pairplot(df, hue='Emission Class' , diag_kind='hist')


# %% [markdown]
# >#### iv) visualize a correlation heatmap between numeric columns
# 

# %%

# select only the numeric columns

df_num = df.select_dtypes(exclude="object")
corr = df_num.corr()

# annot to show the numbers in the cells

sns.heatmap(corr, annot=True)



# %% [markdown]
# ## c) Preprocess the data such that:

# %%
# get copy from the original to preprocess

df_pre = df.copy()

# %% [markdown]
# >#### i) the features and targets are separated

# %%
targets_columns = ['CO2 Emissions(g/km)' , 'Emission Class']

df_targets = df_pre[targets_columns]
df_features = df_pre.drop(columns=targets_columns)

print(df_features.head())
print(df_targets.head())

# %% [markdown]
# >#### ii) categorical features and targets are encoded

# %%

# set the categorical features

label_col_encoded_features = ['Make', 'Model', 'Vehicle Class', 'Transmission' ,'Fuel Type' ]



le = LabelEncoder()
df_features[label_col_encoded_features] = df_features[label_col_encoded_features].apply(le.fit_transform)

df_features.head()




# %%

# get the unique values for targets to show the order of the values
df_targets['Emission Class'].unique()


# scale with ordinal values
ordinal_col_encoded_targets = ['Emission Class']
categories = [
    ['VERY LOW','LOW','MODERATE', 'HIGH' ]
]

oe = OrdinalEncoder(categories=categories)

# get warning here as the slice by this way can cause problems
# df_targets[ordinal_col_encoded_targets] = oe.fit_transform(df_targets[ordinal_col_encoded_targets])

# here the correct way to set the ":" all the rows and the columns that will be transformed
# to slice correctly to Avoid Ambiguity
df_targets.loc[:, ordinal_col_encoded_targets] = oe.fit_transform(df_targets[ordinal_col_encoded_targets])


df_targets.head()

# %% [markdown]
# 
# >>##### Based on the correlation heatmap, select two features to be the independent variables of your model.
# 

# %%


df_train_combined = pd.concat([df_features, df_targets], axis=1)
corr_matrix = df_train_combined.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True , cmap='coolwarm')

# Fuel Consumption Comb (L/100 km) and Cylinders

# %% [markdown]
# >#### iii) the data is shuffled and split into training and testing set
# 

# %%

# make the 80% from the data training set and 20% from the data testing set
# random state to ensure that the split return the same data each run
df_features, df_targets = shuffle(df_features, df_targets, random_state=42)
df_features_train, df_features_test, df_targets_train, df_targets_test = train_test_split(df_features, df_targets, test_size=0.2, random_state=42) 

print(len(df_features_train))
print(len(df_features_test))
print(len(df_targets_train))
print(len(df_targets_test))

# %% [markdown]
# >#### iv) numeric features are scaled
# 

# %%


scaler = RobustScaler()

# the scaler return ndarray

df_features_train = scaler.fit_transform(df_features_train)
df_features_test = scaler.fit_transform(df_features_test)


# convert the ndarray to DataFrame

df_features_train = pd.DataFrame(df_features_train, columns=df_features.columns)
df_features_test = pd.DataFrame(df_features_test, columns=df_features.columns)



print(df_features_train.describe().T)
print(df_features_test.describe().T)




# %%
selected_features = ['Fuel Consumption Comb (L/100 km)' , 'Cylinders']

df_selected_features_train = df_features_train[selected_features]
df_selected_features_test = df_features_test[selected_features]


print(df_selected_features_train.head())
print(df_selected_features_test.head())

# %% [markdown]
# ## **d) Implement linear regression using gradient descent from scratch**

# %% [markdown]
# >#### ***Hypothesis Function***

# %%
def hypothesis(x, theta):
    y_predections = np.dot(x, theta)
    return y_predections

# %% [markdown]
# >#### ***Cost Function***

# %%
def cost_fun(x, y, theta):
    m = len(y)
    err = 0
    for j in range(m):
        err += ((hypothesis(x[j], theta) - y[j]) ** 2)
    
    cost = (1 / (2 * m)) * err
    return cost


# %% [markdown]
# >#### ***Gradient Descent Implementation***

# %%
def gradient_descent(x, y ,theta, alpha, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        temp_theta = np.copy(theta)
        for j in range(len(theta)):
            err = 0
            for e in range(len(x)):
                err += x[e,j] * (hypothesis(x[e], theta) - y[e])
            
            gradient = ((alpha/m) * err)
            temp_theta[j] = theta[j] - gradient[0]    
        
        theta = temp_theta
        costs.append(cost_fun(x, y, theta))
    
    return theta, costs

# %% [markdown]
# >#### ***Gradient Descent Run***

# %%
n = 2 # number of features
theta = np.zeros(n+1) # added 1 for theta0
alpha = 0.03
iterations = 500 
x = df_selected_features_train.to_numpy()
x = np.c_[np.ones((x.shape[0], 1)), x] # added col of ones for theta0, because x0 = 1
y = pd.DataFrame(df_targets_train, columns=df_targets.columns).drop(columns=['Emission Class']).to_numpy().reshape(-1, 1) # reshape to ensure it's a 2d array

theta, costs = gradient_descent(x, y, theta, alpha, iterations)
print("our theta", theta)
for i in range(0, 500, 50): 
    print(costs[i])


# %% [markdown]
# >#### ***R<sup>2</sup> Score***

# %%
x_test = df_selected_features_test.to_numpy()
x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
y_test = pd.DataFrame(df_targets_test, columns=df_targets.columns).drop(columns=['Emission Class']).to_numpy().reshape(-1, 1)
y_predictions = np.array([hypothesis(row, theta) for row in x_test]).reshape(-1, 1) # run the hypothesis function on each row of x to get the array

r2 = r2_score(y_test, y_predictions)  # R2 Score
print("R^2 Score:", r2)

# %% [markdown]
# >#### ***Gradient Descent Cost Function Plot***

# %%
plt.plot(range(iterations), costs)
plt.title("Gradient Descent Cost Function Plot")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

# %% [markdown]
# ## **e) Fit a logistic regression model to the data**

# %% [markdown]
# >#### ***Prepare the Data***
# 

# %%
x_train = df_selected_features_train
x_test = df_selected_features_test

y_train = df_targets_train.drop(columns=["CO2 Emissions(g/km)"])
y_test = df_targets_test.drop(columns=["CO2 Emissions(g/km)"])

y_train = y_train.astype(int)
y_test = y_test.astype(int)


# %% [markdown]
# >#### ***Initialize and Train the Model***

# %%
model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

# convert the y_train and y_test from 2d arrays to 1d arrays
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

model.fit(x_train, y_train)

# %% [markdown]
# >#### ***Make Predictions and Evaluate the Model***

# %%
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')


