Venture Funding with Deep Learning
You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has given you a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.

Instructions:
The steps for this challenge are broken out into the following sections:

Prepare the data for use on a neural network model.

Compile and evaluate a binary classification model using a neural network.

Optimize the neural network model.

Prepare the Data for Use on a Neural Network Model
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), preprocess the dataset so that you can use it to compile and evaluate the neural network model later.

Open the starter code file, and complete the following data preparation steps:

Read the applicants_data.csv file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.

Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.

Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame.

Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.

Note To complete this step, you will employ the Pandas concat() function that was introduced earlier in this course.

Using the preprocessed data, create the features (X) and target (y) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset.

Split the features and target sets into training and testing datasets.

Use scikit-learn's StandardScaler to scale the features data.

Compile and Evaluate a Binary Classification Model Using a Neural Network
Use your knowledge of TensorFlow to design a binary classification deep neural network model. This model should use the dataset’s features to predict whether an Alphabet Soup–funded startup will be successful based on the features in the dataset. Consider the number of inputs before determining the number of layers that your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate your binary classification model to calculate the model’s loss and accuracy.

To do so, complete the following steps:

Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
Hint You can start with a two-layer deep neural network model that uses the relu activation function for both layers.

Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
Hint When fitting the model, start with a small number of epochs, such as 20, 50, or 100.

Evaluate the model using the test data to determine the model’s loss and accuracy.

Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.

Optimize the Neural Network Model
Using your knowledge of TensorFlow and Keras, optimize your model to improve the model's accuracy. Even if you do not successfully achieve a better accuracy, you'll need to demonstrate at least two attempts to optimize the model. You can include these attempts in your existing notebook. Or, you can make copies of the starter notebook in the same folder, rename them, and code each model optimization in a new notebook.

Note You will not lose points if your model does not achieve a high accuracy, as long as you make at least two attempts to optimize the model.

To do so, complete the following steps:

Define at least three new deep neural network models (the original plus 2 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
Rewind Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:

Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.

Add more neurons (nodes) to a hidden layer.

Add more hidden layers.

Use different activation functions for the hidden layers.

Add to or reduce the number of epochs in the training regimen.

After finishing your models, display the accuracy scores achieved by each model, and compare the results.

Save each of your models as an HDF5 file.

# Imports

# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame

# Review the DataFrame

# Review the data types associated with the columns

Step 2: Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.
# Drop the 'EIN' and 'NAME' columns from the DataFrame

# Review the DataFrame

Step 3: Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame.
# Create a list of categorical variables 

# Display the categorical variables list

# Create a OneHotEncoder instance

# Encode the categorcal variables using OneHotEncoder

# Create a DataFrame with the encoded variables

# Review the DataFrame

# Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame

# Define the target set y using the IS_SUCCESSFUL column

# Display a sample of y

# Define features set X by selecting all columns but IS_SUCCESSFUL

# Review the features DataFrame

Step 6: Split the features and target sets into training and testing datasets.
# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1

Step 7: Use scikit-learn's StandardScaler to scale the features data.
# Create a StandardScaler instance

# Fit the scaler to the features training dataset

# Fit the scaler to the features training dataset

Compile and Evaluate a Binary Classification Model Using a Neural Network
Step 1: Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.
Hint You can start with a two-layer deep neural network model that uses the relu activation function for both layers.

# Define the the number of inputs (features) to the model

# Review the number of features

# Define the number of neurons in the output layer
                      
# Define the number of hidden nodes for the first hidden layer

# Review the number hidden nodes in the first layer

# Define the number of hidden nodes for the second hidden layer

# Review the number hidden nodes in the second layer

# Create the Sequential model instance

# Add the first hidden layer

# Add the second hidden layer

# Add the output layer to the model specifying the number of output neurons and activation function

# Display the Sequential model summary

Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_3 (Dense)             (None, 10)                1170      
                                                                 
 dense_4 (Dense)             (None, 8)                 88        
                                                                 
 dense_5 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 1,267
Trainable params: 1,267
Non-trainable params: 0
_________________________________________________________________
Step 2: Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
# Compile the Sequential model

# Fit the model using 50 epochs and the training data

Step 3: Evaluate the model using the test data to determine the model’s loss and accuracy.
# Evaluate the model loss and accuracy metrics using the evaluate method and the test data

# Display the model loss and accuracy results

# Set the model's file path

# Export your model to a HDF5 file

Optimize the neural network model
Step 1: Define at least three new deep neural network models (resulting in the original plus 3 optimization attempts). With each, try to improve on your first model’s predictive accuracy.
Rewind Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:

Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.

Add more neurons (nodes) to a hidden layer.

Add more hidden layers.

Use different activation functions for the hidden layers.

Add to or reduce the number of epochs in the training regimen.

Alternative Model 1
# Define the the number of inputs (features) to the model

# Review the number of features

# Define the number of neurons in the output layer

# Define the number of hidden nodes for the first hidden layer

# Review the number of hidden nodes in the first layer

# Create the Sequential model instance

# First hidden layer

# Output layer

# Check the structure of the model

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_12 (Dense)            (None, 10)                1170      
                                                                 
 dense_13 (Dense)            (None, 1)                 11        
                                                                 
=================================================================
Total params: 1,181
Trainable params: 1,181
Non-trainable params: 0
_________________________________________________________________
# Compile the Sequential model

# Fit the model using 50 epochs and the training data

Alternative Model 2
# Define the the number of inputs (features) to the model

# Review the number of features

# Define the number of neurons in the output layer

# Define the number of hidden nodes for the first hidden layer

# Review the number of hidden nodes in the first layer

# Create the Sequential model instance

# First hidden layer
# YOUR CODE HERE

# Output layer

# Check the structure of the model

Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_10 (Dense)            (None, 5)                 585       
                                                                 
 dense_11 (Dense)            (None, 1)                 6         
                                                                 
 dense_14 (Dense)            (None, 5)                 10        
                                                                 
 dense_15 (Dense)            (None, 1)                 6         
                                                                 
=================================================================
Total params: 607
Trainable params: 607
Non-trainable params: 0
_________________________________________________________________
# Compile the model

# Fit the model

Step 2: After finishing your models, display the accuracy scores achieved by each model, and compare the results.

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data

# Display the model loss and accuracy results

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data

# Display the model loss and accuracy results

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data

# Display the model loss and accuracy results

Step 3: Save each of your alternative models as an HDF5 file.
# Set the file path for the first alternative model

# Export your model to a HDF5 file

# Export your model to a HDF5 file

# Set the file path for the second alternative model

# Export your model to a HDF5 file
