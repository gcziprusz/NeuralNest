# Glossary of Terms
####  [A](#A) | [B](#B) | [C](#C) | [D](#D) | [E](#E) | [F](#F) | [G](#G) | [H](#H) | [I](#I) | [J](#J) | [K](#K) | [L](#L) | [M](#M) | [N](#N) | [O](#O) | [P](#P) | [Q](#Q) | [R](#R) | [S](#S) | [T](#T) | [U](#U) | [V](#V) | [W](#W) | [X](#X) | [Y](#Y) | [Z](#Z)


# A
- Activation function
  - An activation function is a mathematical operation applied to the output of a neuron in a neural network. It determines whether the neuron should be activated or not based on the weighted sum of its inputs. Activation functions introduce non-linearity, allowing neural networks to learn complex patterns and relationships in data. Common activation functions include Sigmoid, Hyperbolic Tangent (tanh), Rectified Linear Unit (ReLU), Leaky ReLU, and Softmax.
- Adam 
  - Adam, short for "Adaptive Moment Estimation," is an optimization algorithm commonly used in machine learning for training deep neural networks. It combines the benefits of two other popular optimization methods, RMSprop and Momentum.
# B
- backpropagation
  - Backpropagation, short for "backward propagation of errors," is a supervised learning algorithm used in the training of artificial neural networks. It involves the iterative adjustment of the network's weights based on the difference between the predicted output and the actual target values. The process starts with a forward pass, where input data is fed through the network to generate predictions. The error is then calculated by comparing these predictions to the actual targets. In the backward pass, the error is propagated backward through the network, and the weights are updated using optimization techniques, such as gradient descent, to minimize the error. Backpropagation enables neural networks to learn and improve their performance by adjusting their internal parameters during the training process.
- Bag of words
  - TBD
- Bayes rule
  - TBD
# C
- convolution
  - convolution is a mathematical way of combining two signals to form a third signal.
- CNN
  - A specialized type of neural network designed for visual data processing, leveraging convolutional layers to automatically learn hierarchical features from images, making it effective for tasks like image classification and object detection.
- Clustering
  - A machine learning technique that involves grouping similar data points together based on certain features or characteristics, facilitating pattern discovery and analysis within datasets.
- cv2 
  - cv2 is a powerful library for working with images in Python.
# D
- data leakage
  - The inadvertent inclusion of information in the training process that should only be available during the testing phase, often resulting in over-optimistic model performance estimates due to the unintentional exposure to future information.
- Decision Trees
  - Tree-like models that represent decisions and their potential consequences, where each node corresponds to a decision based on a specific feature, commonly used in classification and regression tasks to create interpretable and easily understandable models.
- dense layer
  - A fully connected layer in a neural network where each neuron is connected to every neuron in the adjacent layers, allowing for complex pattern recognition and feature learning in various machine learning tasks.
- Deep Neural Networks
  - A sophisticated neural network architecture with multiple hidden layers, enabling the model to automatically learn intricate representations and hierarchies of features from the input data, often used for complex tasks like image recognition, natural language processing, and more.
- differential calculus
  - TBD
# E
- epoch
  - An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. 
# F
- filter 
  - A filter, or kernel, in a CNN is a small matrix of weights that slides over the input data (such as an image), performs element-wise multiplication with the part of the input it is currently on.
- fittness landscape
  - TBD
# G
- Generative model
  - TBD
- genetic programming
  - TBD
- Gini index
  - TBD
- Gradient Descent
  - TBD
# H
- H20
  - TBD
- hyper parameters
  - TBD
# I
# J
- jupyter
  - Jupyter is an open-source project that provides a web-based interactive computing platform. It supports various programming languages, but it's most commonly used with languages like Python, R, and Julia.
  - The name "Jupyter" is derived from the three core programming languages it initially supported: Julia, Python, and R.
# K
- keras
  - Keras is a high-level neural network library that runs on top of TensorFlow.
- kernel
  - A filter is a collection of kernels, although the ml domain may use the two terms filter and kernel interchangeably.
# L
- Linear Regression 
  - TBD
- learnability
  - TBD
- Logistic Regression
  - TBD
- Loss Function
  - TBD

# M
- Maximum Likelihood Estimation (MLE)
 - It is a statistical method used to estimate the parameters of a probability distribution based on observed data. The goal of MLE is to find the set of parameter values that maximize the likelihood function, which measures the probability of observing the given data under the assumed probability distribution.
- Multiple Regression 
  - Multiple regression is a statistical technique used to analyze the relationship between a dependent variable and two or more independent variables. It extends the concept of simple linear regression, which involves predicting a dependent variable based on a single independent variable, to cases where there are multiple predictors.
- MLP 
  - MLP stands for Multilayer Perceptron, which is a type of artificial neural network (ANN) composed of multiple layers of neurons (nodes) arranged in a feedforward manner. MLPs are one of the most common types of neural networks used in machine learning and are widely applied to various tasks, including classification, regression, and pattern recognition.
- MNIST 
  - MNIST stands for Modified National Institute of Standards and Technology database, and it's a widely used dataset in the field of machine learning and computer vision. It consists of a large collection of handwritten digits that are commonly used for training various image classification algorithms.
# N
- Naive bayes
  - Naive Bayes is a classification algorithm based on Bayes' theorem with an assumption of independence among predictors (features). It's a simple yet effective algorithm commonly used for text classification and other classification tasks in machine learning.
- NEAT
  - NEAT, which stands for NeuroEvolution of Augmenting Topologies, is a genetic algorithm-based approach.
- NumPy
  - NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
# O
- One-hot encoding 
  - It is a technique used to represent categorical variables as binary vectors. In the context of machine learning, it's often employed to represent categorical labels in a format that's suitable for training models. 
  For example, consider a categorical variable representing fruit types: {apple, banana, cherry}. The one-hot encoding would be:

    ```
    apple: [1, 0, 0]
    banana: [0, 1, 0]
    cherry: [0, 0, 1]
    ```
- optimizer 
  - An optimizer in the context of machine learning refers to an algorithm or method used to adjust the parameters of a model during training to minimize the error or loss function. The primary goal of optimization is to find the optimal set of parameters that result in the best performance of the model on the training data.
- overfitting
  - Overfitting is a common issue in machine learning where a model learns the training data too well, capturing noise and random fluctuations instead of the underlying patterns. As a result, an overfit model performs exceptionally well on the training data but fails to generalize effectively to new, unseen data.
# P
- pandas
  - Pandas and NumPy are two popular Python libraries used for data manipulation and analysis.
    Pandas is designed for data manipulation and analysis, especially for working with labeled and relational data. Pandas provides two primary data structures: Series (1-dimensional labeled array) and DataFrame (2-dimensional labeled data structure, like a table in a database). Use Pandas for data manipulation, cleaning, and analysis, especially when dealing with labeled or tabular data.
- Principal Component Analysis
  - TBD
- Perceptrons
  - TBD
- probibilistic context free grammar
  - TBD 
- pooling
  - TBD
- Plotly
  - TBD
- PyTorch
  - TBD

# Q
# R
- Recommendation Systems
  - TBD 
- relu (rectified linear unit)
  - TBD 

# S
- Scikit Learn
  - TBD
- Seaborn
  - TBD 
- Support Vector Machines
  - TBD 
- stochastic gradient descent (sgd)
  - SGD is an iterative optimization algorithm that aims to minimize a cost function by updating the model parameters in the opposite direction of the gradient of the cost function. 
- Sparse Categorical Crossentropy (SCC)
  - Sparse Categorical Crossentropy (SCC) is a loss function used in multiclass classification tasks where the target labels are integers (sparse representation) rather than one-hot encoded vectors. It is commonly used in machine learning frameworks like TensorFlow and Keras.
- Sigmoid function 
  - The sigmoid function is a mathematical function that maps input values to a smooth S-shaped curve, typically ranging from 0 to 1 or -1 to 1. In machine learning, the sigmoid function is commonly used as an activation function in neural networks, particularly in the output layer for binary classification problems. It is also used in logistic regression models to model binary outcomes and in other contexts where the output needs to be bounded between 0 and 1.
- Spyder
  - A free and open source scientific environment written in Python.
- symbolic regression
  - Symbolic regression is a machine learning technique that aims to find mathematical expressions or formulas that best describe a given dataset. Unlike traditional regression methods that fit parametric models (such as linear regression or polynomial regression) to data, symbolic regression seeks to discover mathematical relationships in an unrestricted, symbolic form.
# T
- tableau
  - Tableau is a powerful data visualization software that allows users to create interactive and shareable dashboards, reports, and visualizations from various data sources. Developed by Tableau Software, now a part of Salesforce, Tableau provides a user-friendly interface for analyzing and presenting data in a visually compelling manner.
- Tensor
  - A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array. A vector is a one dimensional or first order tensor and a matrix is a two dimensional or second order tensor.
- tensorflow
  - TensorFlow is an end-to-end open source platform for machine learning.
- TFIDF (Term Frequency-Inverse Document Frequency)
  -  It's a numerical statistic used in information retrieval and text mining to evaluate the importance of a word in a document relative to a collection of documents (corpus). TF-IDF is commonly used as a feature extraction technique in natural language processing (NLP) tasks, such as document classification, clustering, and search engine ranking. (TF): Term frequency measures how frequently a term (word) occurs in a document. (IDF): Inverse document frequency measures the importance of a term across the entire corpus. It's calculated by dividing the total number of documents in the corpus by the number of documents containing the term, and then taking the logarithm of the result. Terms that occur in many documents are given lower weights, while terms that occur in fewer documents are given higher weights.
- trigonometric functions (cos ...)
  - Trigonometric functions are mathematical functions that relate the angles of a right triangle to the ratios of the lengths of its sides. In machine learning, trigonometric functions can be utilized in various ways, especially when dealing with problems involving periodic patterns or cyclic data.
- tree bank
  - A "treebank" refers to a type of corpus or dataset used in natural language processing (NLP) and computational linguistics for syntactic analysis and parsing. It is a collection of sentences annotated with syntactic structures, typically represented as parse trees. Treebanks are essential resources in NLP research and development, as they enable the development and evaluation of syntactic analysis algorithms and models. They are often created manually by linguists or annotated using automated tools and are available for various languages and domains. Well-known treebanks include the Penn Treebank for English, the Universal Dependencies project, and the Prague Dependency Treebank for Czech.
# U
- Underfitting 
  - Underfitting occurs when a machine learning model is too simple to capture the underlying structure of the data. In other words, the model is not able to learn the training data effectively, leading to poor performance on both the training data and new, unseen data. Underfitting typically happens when the model is too simple or lacks the capacity to represent the complexity of the underlying data distribution.

# V
# W
# X
# Y
# Z
