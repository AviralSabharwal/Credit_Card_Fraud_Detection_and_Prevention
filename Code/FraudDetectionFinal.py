# Fraud Detection using Self Organizing Maps

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Importing the dataset and Analyzing it
data = pd.read_csv('Credit_Card_Applications.csv')
data = data.drop(["CustomerID"],axis = 1)

# Analysis 1: Outliers Detection and class imbalance
ContinuousData = pd.DataFrame()
ContinuousVariableList = ["A2", "A3", "A7", "A10", "A13", "A14"]
for var in ContinuousVariableList:
    ContinuousData[var] = data[var].astype("float32")
fig, axes = plt.subplots(nrows = 1,ncols = 1)
fig.set_size_inches(10, 10)
sn.boxplot(data = ContinuousData, orient = "V", ax = axes)

# Analysis 2: Correlation Analysis
CorrelationMatrix = data.corr()
mask = np.array(CorrelationMatrix)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sn.heatmap(CorrelationMatrix, mask = mask, vmax = 0.8, square = True, annot = True) 

# Importing the dataset again to work on it
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM

# We define these further up in the code when creating the SOM
rows = 10
cols = 10

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5, random_seed = 100)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the Results
from pylab import bone, pcolor, colorbar, plot, show
plt.figure()
bone()
pcolor(som.distance_map().T, cmap = "winter")
colorbar()
markers = ['o', '.']
colors = ['w', 'k']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()    

# Get the customers that are likely to defraud the bank (potential cheaters)
# Add indices to SOM values & sort by value
helper = np.concatenate(
    (som.distance_map().reshape(rows*cols, 1),         # the SOM map values
     np.arange(rows*cols).reshape(rows*cols, 1)),      # concatenated with an index
    axis=1)                                            # as a 2D matrix with 2 columns of data
helper = helper[helper[:, 0].argsort()][::-1]          # sort by first column (map values) and reverse (so top values are first)
# First we choose how many cells to take as outliers...
use_threshold = True   # toggle usage for calculating indices (pick cells that exceed threshold or use hardcoded number of cells)
top_cells = 4     # 4 out of 100 seems a valid idea, but ideally it might be chosen after inspecting the current SOM plot
threshold = 0.8      # Use threshold to select top cells
# Take indices that correspond to cells we're interested in
idx = helper[helper[:, 0] > threshold, 1] if use_threshold else helper[:top_cells, 1]
# Find the data entries assigned to cells corresponding to the selected indices
result_map = []
mappings = som.win_map(X)
for i in range(rows):
    for j in range(cols):
        if (i*rows+j) in idx:
            if len(result_map) == 0:                
                result_map = mappings[(i,j)]
            else:
                # Sometimes a cell contains no observations (customers)
                # This will cause numpy to raise an exception so to avoid it...
                if len(mappings[(i,j)]) > 0:
                    result_map = np.concatenate((result_map, mappings[(i,j)]), axis=0)                
 
# finally we get our fraudster candidates
frauds = sc.inverse_transform(result_map)
    
# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]

# Primary Fraud List
PrimaryFraudList = np.concatenate(([is_fraud], [y]), axis =0)
PrimaryFraudList = PrimaryFraudList.T
         
# This is the list of potential cheaters (customer ids)
print(frauds[:, 0]) 
