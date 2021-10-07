# Neural_Networks_API

## Implemented:
1. Stochastic Gradient Descent
2. Batch Gradient Descent
3. Momentum Based Gradient Descent
4. Nesterov's accelerated gradient descent
## Loss Functions:
1. Mean Squared Error
2. Categorical Cross_entropy
## Neuron Types:
1. Sigmoid_Layer
2. Softmax_Layer
3. Relu_Layer
## Libraries used:
1. numpy
2. pandas
3. matplotlib
4. using pickle is optional: It is just used for saving the model object
## Documentation:
- Layers:
  ``` python
  model.add_Dense(neuron_type,num_neurons)      # neuron_type=Sigmoid_Layer, Softmax_Layer
  ```
- Training:
  ``` python
  model.fit(x_train,y_train,x_test,y_test,n_epoch,batch_size)       # used for training the model
  ```
- Validation:
  ``` python
  model.validate(x,y)       # to find the accuracy on validation dataset
  ```
- Data Preparation: 
  ``` python
  get_data(path)            # to seperate the feature vectors and labels from the csv file
  get_test_data(path)       # to get the test data
  get_submission_csv(path,model_obj,submission_file_path)       # Get a kaggle submission file for the given test file (path) 
  ```
- Test Train Split:
  ``` python
  test_train_split(frac,x,y)        # to get the train and test data in the required fracion
  ``` 
- Predict:
  ``` python
  get_predictions(dataset, model_obj)       # dataset==the test_dataset, object of the model class used
  ```
## Results:
- Sigmoid in the last layer does not work well with the categorical cross entropy loss function.
- Small Dataset requires a small neural network so 3 layers seem to be enough.
- Note: Accuracy is not remaining the same for a neural network trained again.
- Softmax with categorical cross entropy gives the best accuracy.
*Links of various runs can be found along with their names in the wandb database.*
## Points to Note:
1. Sigmoid layer at the output is performing very good as compared to softmax layer.
2. Momentum based gradient descent with moment (0.1) gets same accuracy as plain model(90, 88) in just 20 epochs as against 40 epochs of plain model with error of 0.0094 and gave 92.5% accuracy on kaggle with 40 epochs.
3. True divide does not work in a softmax function while np.divide() works.....why?
4. Stochastic works better than batch gradient descent
5. Reducing 1 layer performs better but the model still suffers from bias


# Density Based clustering of applications with Noise:
- Libraries used: numpy, pandas, matplotlib, sklearn (for creating datasets)
# Class Dbscan
- Fit the data set and train the clustering algorithm:
  ``` python
  model.fit(list_of_training_data_points)
  ```
- Return a list of all the neighbors of the point satisfying the epsilon conditions
  ``` python
  model.get_neighbors(pt_whose_neighbors_are_to_be_found)
  ```
- Return the number of clusters identified so far
  ``` python
  model.number_of_clusters()
  ```
- Class Point: contains 2 attributes: 
  ``` python
  pt.value        # gives the coordinate vector of the point
  pt.cluster      # gives the cluster number to which the point belongs
  ```
  
*The algorithm above works on the depth first or breadth first search in a
graph where it finds the number of disjoint graphs and clusters out all the
points that do not satisfy the conditions to be core points or edge
points. These points are thus classified as Noise Points.*


# KMeans clustering with anomalous initialization
- Libraries used: numpy, matplotlib, sklearn (for datasets), pandas
- Used Iris data set to compare the Kmeans from sklearn to my implementation
# Class KMeans_
- Train:
  ``` python
  model.fit(X)      # X - list of training data points
  ```
- Return the centers of various clusters identified
  ``` python
  model.cluster_centers()
  ```
- Return a list of the predicted labels for the datapoints
  ``` python
  model.labels(X)   # X - list of data points to predict upon 
  ```
- Return the list of all the elements in the specified cluster
  ``` python
  model.get_cluster_elements(X,cluster_number)
  ```
  
# How is it different from vanilla KMeans clustering?
- Initialization is not random. It is done following the structure of data.
- minkowski’s distance metric has been used rather than euclidean distance to include multiple weight metrics.
- Weighted Kmeans has been used which gives more preference to important features while less preference to less important features in a feature vector.
