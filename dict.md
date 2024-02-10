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
  - Bag of Words (BoW) is a foundational technique in natural language processing (NLP) for representing text data in a structured format suitable for machine learning algorithms. It involves breaking down a piece of text into individual words (or tokens) and creating a numerical representation based on the frequency of occurrence of these words within the text.
- Bayes rule
  - Bayes' Rule, also known as Bayes' Theorem or Bayes' Law, is a fundamental principle in probability theory named after the Reverend Thomas Bayes. It provides a mathematical framework for updating the probability of a hypothesis in light of new evidence.
- Binary Cross-Entropy Loss
  - TBD
# C
- Categorical Cross-Entropy Loss
  - TBD
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
- dot product 
  - The dot product, also known as the scalar product or inner product, is a mathematical operation that takes two equal-length sequences of numbers (usually vectors) and returns a single scalar value. It is denoted by a dot (.) or sometimes by simply placing the vectors side by side without any operator.
- differential calculus
  - Differential calculus is a branch of calculus that focuses on studying the rates at which quantities change. It deals with concepts such as derivatives and rates of change. Differential calculus is fundamental in many fields of science, engineering, economics, and more, as it provides tools for understanding how quantities behave and how they are related to each other.
# E
- epoch
  - An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. 
# F
- feature engineering 
  - Feature engineering is a crucial process in machine learning and data science that involves selecting, transforming, and creating input variables (features) to improve the performance of predictive models. It aims to enhance the quality and relevance of the features used by the model, thereby improving its ability to make accurate predictions or classifications.
- filter 
  - A filter, or kernel, in a CNN is a small matrix of weights that slides over the input data (such as an image), performs element-wise multiplication with the part of the input it is currently on.
- fittness landscape
  - In machine learning, a fitness landscape is a conceptual model used to represent the relationship between different sets of model parameters (or solutions) and their corresponding performance metrics (fitness). It provides a way to visualize and understand how the performance of a machine learning model changes as its parameters vary.
# G
- Generative model
  - 
A generative model is a type of model in machine learning that learns the underlying probability distribution of the training data in order to generate new data samples that are similar to the training data. Generative models are used to model the joint probability distribution P(X,Y) of input features X and target labels Y, or just the distribution P(X) of input features.

- genetic programming
  - Genetic programming (GP) is a type of evolutionary algorithm used in machine learning and optimization. It is inspired by the process of natural selection and Darwinian evolution. Genetic programming evolves computer programs, represented as hierarchical structures (usually trees), to solve a specific problem or perform a given task.

- Gini index
  - The Gini index, also known as the Gini impurity, is a measure of the impurity or disorder of a set of data points in the context of classification problems. It is commonly used in decision tree algorithms, such as CART (Classification and Regression Trees), to evaluate the quality of a split at a given node.

- Gradient Descent
  - Gradient descent is an optimization algorithm used to minimize the cost function or loss function in machine learning and deep learning models. It is a first-order iterative optimization algorithm that moves towards the minimum of the function by taking steps proportional to the negative of the gradient of the function at the current point.
# H
- Hinge Loss
  - TBD
- Huber Loss
  - TBD
- H20
  - H2O is an open-source, distributed machine learning platform designed for big data analytics and machine learning tasks. It provides scalable and efficient implementations of popular machine learning algorithms, enabling users to perform advanced analytics on large datasets using distributed computing techniques.
- hyper parameters
  - Hyperparameters are parameters that are not directly learned from the training data but are set prior to the training process. They control the behavior and performance of machine learning algorithms and models and are typically tuned during the training process to optimize the model's performance on the validation or test data.
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
- l2 norm
  - The L2 norm, also known as the Euclidean norm or the Euclidean distance, is a measure of the magnitude of a vector in Euclidean space. It is defined as the square root of the sum of the squared absolute values of the components of the vector. 
- Linear Regression 
  - Linear regression is a statistical method used to model the relationship between a dependent variable (often denoted as 
(Y) and one or more independent variables (often denoted as 
(X). It assumes a linear relationship between the independent variables and the dependent variable. Linear regression is widely used for prediction and inference tasks in various fields, including economics, finance, engineering, and social sciences.
- learnability
  - In the context of machine learning, learnability refers to the theoretical and practical aspects of whether a particular problem or concept can be learned from data. It encompasses the questions of whether a given learning algorithm can effectively learn a target function from training data and whether the learned model can generalize well to unseen data.
- Logistic Regression
  - Logistic regression is a statistical method used for modeling the relationship between a binary dependent variable (target) and one or more independent variables (features). Despite its name, logistic regression is a classification algorithm rather than a regression algorithm, and it's widely used for binary classification tasks.
- Loss Function
  - A loss function, also known as a cost function or objective function, is a measure of how well a machine learning model is performing with respect to its training data and the task it is trying to accomplish. It quantifies the difference between the predicted output of the model and the actual target values in the training data. The choice of a loss function depends on the type of machine learning task being performed (e.g., regression, classification) and the specific characteristics of the problem domain. Some common types of loss functions include:

# M
- manhattan distance
  - Manhattan distance, also known as city block distance or distance, is a metric used to measure the distance between two points in a grid-based system. It is named "Manhattan distance" because it measures the distance a person would walk along the streets of a city laid out in a grid-like pattern, where they can only move horizontally and vertically (no diagonal movement).
- Maximum Likelihood Estimation (MLE)
 - It is a statistical method used to estimate the parameters of a probability distribution based on observed data. The goal of MLE is to find the set of parameter values that maximize the likelihood function, which measures the probability of observing the given data under the assumed probability distribution.
- Mean Squared Error (MSE):
  - TBD
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
  - Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving most of the important information. It achieves this by identifying the principal components, which are the directions in which the data varies the most.
- Perceptrons
  - Perceptrons are the building blocks of artificial neural networks and serve as basic computational units that mimic the functioning of biological neurons. They were introduced by Frank Rosenblatt in 1957 and are one of the earliest forms of artificial neural networks.
- probibilistic context free grammar
  - A Probabilistic Context-Free Grammar (PCFG) is an extension of a context-free grammar (CFG) in which each production rule is associated with a probability. PCFGs are used in natural language processing (NLP) and computational linguistics to model the structure of sentences and generate syntactically valid sentences with associated probabilities. 
- pooling
  - Pooling is a technique used in convolutional neural networks (CNNs) to reduce the spatial dimensions (width and height) of feature maps while retaining the most important information. It helps in controlling overfitting, reducing computational complexity, and improving the efficiency of the network.
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
