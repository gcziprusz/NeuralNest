# Glossary of Terms
####  [A](#A) | [B](#B) | [C](#C) | [D](#D) | [E](#E) | [F](#F) | [G](#G) | [H](#H) | [I](#I) | [J](#J) | [K](#K) | [L](#L) | [M](#M) | [N](#N) | [O](#O) | [P](#P) | [Q](#Q) | [R](#R) | [S](#S) | [T](#T) | [U](#U) | [V](#V) | [W](#W) | [X](#X) | [Y](#Y) | [Z](#Z)


# A
- adam
    - Adam (short for Adaptive Moment Estimation) is an optimization algorithm commonly used for training deep learning models. It is an extension of stochastic gradient descent (SGD) and combines ideas from both momentum optimization and RMSprop (Root Mean Square Propagation).
- adversarial example
    - An adversarial example refers to a carefully crafted input to a machine learning model that is intentionally designed to cause the model to make a mistake or misclassify the input. These examples are created by making small, imperceptible perturbations to the input data, which are often imperceptible to humans but can significantly affect the model's predictions.
- automatic speech recognition
    - Automatic Speech Recognition (ASR), also known as speech-to-text conversion, is the technology that enables computers to transcribe spoken language into text. ASR systems process audio input from various sources, such as microphones or recorded audio files, and convert it into textual form, making it accessible for further analysis, processing, or integration into other applications.
- Activation function
  - An activation function is a mathematical operation applied to the output of a neuron in a neural network. It determines whether the neuron should be activated or not based on the weighted sum of its inputs. Activation functions introduce non-linearity, allowing neural networks to learn complex patterns and relationships in data. Common activation functions include Sigmoid, Hyperbolic Tangent (tanh), Rectified Linear Unit (ReLU), Leaky ReLU, and Softmax.
- Adam 
  - Adam, short for "Adaptive Moment Estimation," is an optimization algorithm commonly used in machine learning for training deep neural networks. It combines the benefits of two other popular optimization methods, RMSprop and Momentum.
# B
- bach normalization
    - Batch Normalization (BatchNorm) is a technique used to accelerate the training of deep neural networks by normalizing the input of each layer. It was proposed by Sergey Ioffe and Christian Szegedy in their paper titled "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
- backpropagation
  - Backpropagation, short for "backward propagation of errors," is a supervised learning algorithm used in the training of artificial neural networks. It involves the iterative adjustment of the network's weights based on the difference between the predicted output and the actual target values. The process starts with a forward pass, where input data is fed through the network to generate predictions. The error is then calculated by comparing these predictions to the actual targets. In the backward pass, the error is propagated backward through the network, and the weights are updated using optimization techniques, such as gradient descent, to minimize the error. Backpropagation enables neural networks to learn and improve their performance by adjusting their internal parameters during the training process.
- Bag of words
  - Bag of Words (BoW) is a foundational technique in natural language processing (NLP) for representing text data in a structured format suitable for machine learning algorithms. It involves breaking down a piece of text into individual words (or tokens) and creating a numerical representation based on the frequency of occurrence of these words within the text.
- Bayes rule
  - Bayes' Rule, also known as Bayes' Theorem or Bayes' Law, is a fundamental principle in probability theory named after the Reverend Thomas Bayes. It provides a mathematical framework for updating the probability of a hypothesis in light of new evidence.
- Binary Cross-Entropy Loss
  - Binary Cross-Entropy Loss, also known as Binary Log Loss, is a common loss function used in binary classification tasks, particularly in scenarios where the model outputs probabilities of belonging to one of the two classes. It measures the difference between the predicted probabilities and the true binary labels.
# C
- causal inference
    - TBD
- causal graph
    - TBD
- convolution
    - Convolution is a fundamental mathematical operation widely used in various fields, including signal processing, image processing, and deep learning. In the context of deep learning, convolutional operations play a central role in convolutional neural networks (CNNs), which are especially well-suited for tasks involving grid-like data, such as images
- convolutional layer
    - Convolutional layers apply convolution operations to input data using learnable filters, allowing the network to automatically extract hierarchical representations of the input.
- Categorical Cross-Entropy Loss
  - Categorical Cross-Entropy Loss, also known as Softmax Cross-Entropy Loss, is a widely used loss function in multi-class classification tasks. It measures the difference between the predicted class probabilities and the true class labels in scenarios where the target variable can belong to one of multiple classes.
- Counterfactual regret 
  - It is a concept used in the context of game theory and specifically in the analysis of strategies in extensive-form games with imperfect information, such as poker. It is a measure of how much a player regrets not having played a different strategy after observing the outcome of the game.
- convolution
  - convolution is a mathematical way of combining two signals to form a third signal.
- CNN
  - A specialized type of neural network designed for visual data processing, leveraging convolutional layers to automatically learn hierarchical features from images, making it effective for tasks like image classification and object detection.
- cross validation
  - Cross-validation is a resampling technique used to evaluate the performance of a machine learning model on unseen data. It is particularly useful for assessing how well a model generalizes to new data and for selecting optimal hyperparameters.
- Contextual Bandit Problem
  - The Contextual Bandit Problem is an extension of the classical multi-armed bandit problem that incorporates additional contextual information or features into the decision-making process. In the contextual bandit problem, each action (or arm) is associated with a set of features or context, and the goal is to learn a policy that selects the best action based on the observed context while maximizing cumulative rewards.
- Clustering
  - A machine learning technique that involves grouping similar data points together based on certain features or characteristics, facilitating pattern discovery and analysis within datasets.
- Classification
  - Classification is a supervised learning task where the goal is to predict the categorical class label of new instances based on past observations. The output variable in classification is discrete and represents a category or class.
- cv2 
  - cv2 is a powerful library for working with images in Python.
# D
- dowhy by microsoft 
    - TBD
- do calculus
    - TBD
- derivative
    - In calculus, the derivative of a function represents the rate at which the function's value changes with respect to its input variable. It provides information about how the function behaves locally around a specific point, such as whether it is increasing, decreasing, or stationary at that point.
- differentiable programming
    - Differentiable programming is a way of writing computer programs that makes them easy to differentiate. In other words, you can easily find out how changing one part of the program will affect the output, even if the program is complex.
- diffusion model
    - The diffusion model is a computational model commonly used in cognitive psychology and neuroscience to describe decision-making processes. It is based on the idea that decision-making involves accumulating evidence over time until a threshold is reached, at which point a decision is made. In machine learning, the diffusion model refers to a probabilistic generative model used for modeling sequential data, such as time series or text data. Unlike the diffusion models used in cognitive psychology, which focus on decision-making processes, the diffusion model in machine learning is primarily used for density estimation and generative modeling tasks.
- DQN
    - DQN stands for Deep Q-Network, which is a type of reinforcement learning algorithm used for training agents to make sequential decisions in environments with discrete actions. It was introduced by DeepMind in 2013 and has since become a foundational algorithm in the field of deep reinforcement learning.
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
- expectation maximization
    - Expectation-Maximization (EM) is a powerful iterative algorithm used for estimating parameters of probabilistic models, particularly when dealing with latent variables or missing data. It is widely applied in various fields, including machine learning, statistics, and signal processing, for tasks such as clustering, density estimation, and parameter estimation in mixture models.
- Euclidean distance 
  - It's a measure of the straight-line distance between two points in Euclidean space. It is the most common metric used to calculate the distance between two points in a two- or multi-dimensional space.
- epoch
  - An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. 

- Epsilon-Greedy
  - Epsilon-Greedy is a simple yet effective strategy used in reinforcement learning and multi-armed bandit problems to balance exploration and exploitation. The term "epsilon" refers to a small positive value (usually between 0 and 1), representing the exploration rate.
- Exploration vs. Exploitation 
  - In reinforcement learning and bandit problems, there is a trade-off between exploration (trying new actions to learn their rewards) and exploitation (choosing actions that are known to have high rewards). Epsilon-Greedy aims to balance these two objectives
# F
- fairsec
    - FairSec, short for "Fair Sequence-to-Sequence," is a research framework developed by Facebook AI Research (FAIR) that aims to address fairness issues in machine translation systems. Machine translation systems, which convert text from one language to another, often suffer from biases that can lead to unfair or inequitable translations, especially for underrepresented languages or sensitive topics.
- feature engineering 
  - Feature engineering is a crucial process in machine learning and data science that involves selecting, transforming, and creating input variables (features) to improve the performance of predictive models. It aims to enhance the quality and relevance of the features used by the model, thereby improving its ability to make accurate predictions or classifications.
- filter 
  - A filter, or kernel, in a CNN is a small matrix of weights that slides over the input data (such as an image), performs element-wise multiplication with the part of the input it is currently on.
- fittness landscape
  - In machine learning, a fitness landscape is a conceptual model used to represent the relationship between different sets of model parameters (or solutions) and their corresponding performance metrics (fitness). It provides a way to visualize and understand how the performance of a machine learning model changes as its parameters vary.
- Foundation Model (AWS)
 - A Model trained on massive datasets, foundation models (FMs) are large deep learning neural networks that have changed the way data scientists approach machine learning (ML). Rather than develop artificial intelligence (AI) from scratch, data scientists use a foundation model as a starting point to develop ML models that power new applications more quickly and cost-effectively. The term foundation model was coined by researchers to describe ML models trained on a broad spectrum of generalized and unlabeled data and capable of performing a wide variety of general tasks such as understanding language, generating text and images, and conversing in natural language.
# G
- glove
    - GloVe, short for Global Vectors for Word Representation, is an unsupervised learning algorithm used to obtain vector representations (embeddings) for words in a high-dimensional vector space. These word embeddings capture semantic and syntactic relationships between words based on their co-occurrence statistics in large text corpora.
- gram matrix
    - The Gram matrix, also known as the Gramian matrix or the covariance matrix, is a mathematical concept commonly used in linear algebra, signal processing, and machine learning. It is named after the Danish mathematician Jørgen Pedersen Gram. In the context of machine learning and deep learning, the Gram matrix is often used in feature extraction and style transfer tasks, particularly in the field of computer vision.
- Generative Adversarial Networks (GANs)
  - Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms introduced by Ian Goodfellow and his colleagues in 2014. GANs are composed of two neural networks, namely the generator and the discriminator, which are trained simultaneously through a competitive process.
- Generative model
  - A generative model is a type of model in machine learning that learns the underlying probability distribution of the training data in order to generate new data samples that are similar to the training data. Generative models are used to model the joint probability distribution P(X,Y) of input features X and target labels Y, or just the distribution P(X) of input features.

- genetic programming
  - Genetic programming (GP) is a type of evolutionary algorithm used in machine learning and optimization. It is inspired by the process of natural selection and Darwinian evolution. Genetic programming evolves computer programs, represented as hierarchical structures (usually trees), to solve a specific problem or perform a given task.

- Gini index
  - The Gini index, also known as the Gini impurity, is a measure of the impurity or disorder of a set of data points in the context of classification problems. It is commonly used in decision tree algorithms, such as CART (Classification and Regression Trees), to evaluate the quality of a split at a given node.

- Gradient Descent
  - Gradient descent is an optimization algorithm used to minimize the cost function or loss function in machine learning and deep learning models. It is a first-order iterative optimization algorithm that moves towards the minimum of the function by taking steps proportional to the negative of the gradient of the function at the current point.
# H
- hidden markov models hmms
    - TBD
- Hallucinations
  - AI hallucinations, also known as AI-generated hallucinations or hallucinatory AI, refer to instances where artificial intelligence systems produce outputs that resemble hallucinations experienced by humans. These hallucinations can occur in various AI-generated content such as images, text, or even audio.
- Hidden Markov Models (HMMs) 
  - They are probabilistic models used to model sequences of observable events or data that are assumed to be generated by an underlying unobservable (hidden) Markov process. HMMs are widely used in various fields, including speech recognition, bioinformatics, natural language processing, and finance.
- Hinge Loss
  - Hinge loss is a loss function commonly used in machine learning, particularly in the context of classification tasks with support vector machines (SVMs) and other linear classifiers. It is designed to measure the loss incurred by a model based on the margins of misclassified examples.
- Huber Loss
  - Huber loss, also known as smooth L1 loss, is a loss function commonly used in regression tasks, particularly in scenarios where the data may contain outliers or noise. It combines the best properties of mean squared error (MSE) loss and mean absolute error (MAE) loss by being quadratic for small errors and linear for large errors. This makes Huber loss less sensitive to outliers compared to MSE loss while still providing smooth gradients for optimization.
- H20
  - H2O is an open-source, distributed machine learning platform designed for big data analytics and machine learning tasks. It provides scalable and efficient implementations of popular machine learning algorithms, enabling users to perform advanced analytics on large datasets using distributed computing techniques.
- hyper parameters
  - Hyperparameters are parameters that are not directly learned from the training data but are set prior to the training process. They control the behavior and performance of machine learning algorithms and models and are typically tuned during the training process to optimize the model's performance on the validation or test data.
# I
- inverse reinforcement
    - TBD
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
- latent semantic analisys
    - TBD
- Laplace smoothing
  - Laplace smoothing, also known as add-one smoothing or additive smoothing, is a technique used to smooth probability estimates in cases where some outcomes have zero probabilities in the training data. It is commonly applied in Bayesian statistics, natural language processing, and machine learning algorithms, particularly in the context of naive Bayes classifiers and language models.
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
- Loss Function (Cost Function)
  - A loss function, also known as a cost function or objective function, is a measure of how well a machine learning model is performing with respect to its training data and the task it is trying to accomplish. It quantifies the difference between the predicted output of the model and the actual target values in the training data. The choice of a loss function depends on the type of machine learning task being performed (e.g., regression, classification) and the specific characteristics of the problem domain. Some common types of loss functions include:

# M
- max pooling layer
    - TBD
- manhattan distance
  - Manhattan distance, also known as city block distance or distance, is a metric used to measure the distance between two points in a grid-based system. It is named "Manhattan distance" because it measures the distance a person would walk along the streets of a city laid out in a grid-like pattern, where they can only move horizontally and vertically (no diagonal movement).
- Maximum Likelihood Estimation (MLE)
 - It is a statistical method used to estimate the parameters of a probability distribution based on observed data. The goal of MLE is to find the set of parameter values that maximize the likelihood function, which measures the probability of observing the given data under the assumed probability distribution.
- Maximum margin classifier
  - The maximum margin classifier, often associated with support vector machines (SVMs), is a binary linear classification model that aims to find the hyperplane that maximizes the margin between classes in the feature space. It is a powerful algorithm for classification tasks, particularly in scenarios where the data is linearly separable or nearly separable.
- Mean Squared Error (MSE)
  - Mean Squared Error (MSE) is a commonly used loss function in regression tasks. It measures the average squared difference between the predicted values and the actual values in a dataset. MSE is widely used because it penalizes large errors more heavily than small errors, making it suitable for tasks where outliers or large deviations from the true values are important.
  ```
  𝐽(𝑤,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2(1)
  ```
- Multiple Regression 
  - Multiple regression is a statistical technique used to analyze the relationship between a dependent variable and two or more independent variables. It extends the concept of simple linear regression, which involves predicting a dependent variable based on a single independent variable, to cases where there are multiple predictors.
- multi-armed bandit problem
  - The multi-armed bandit problem is a classic problem in decision theory and sequential decision-making under uncertainty. It is named after the metaphor of a gambler facing a row of slot machines (bandits), each with a different probability distribution of payouts. The goal of the gambler is to maximize the total reward gained over time by choosing which slot machine to play in each round.
- MLP 
  - MLP stands for Multilayer Perceptron, which is a type of artificial neural network (ANN) composed of multiple layers of neurons (nodes) arranged in a feedforward manner. MLPs are one of the most common types of neural networks used in machine learning and are widely applied to various tasks, including classification, regression, and pattern recognition.
- MNIST 
  - MNIST stands for Modified National Institute of Standards and Technology database, and it's a widely used dataset in the field of machine learning and computer vision. It consists of a large collection of handwritten digits that are commonly used for training various image classification algorithms.
# N
- nuisance variables
    - TBD
- Naive bayes
  - Naive Bayes is a classification algorithm based on Bayes' theorem with an assumption of independence among predictors (features). It's a simple yet effective algorithm commonly used for text classification and other classification tasks in machine learning.
- Nash equilibrium 
  - It is a fundamental concept in game theory, named after the mathematician and economist John Nash. It refers to a situation in which each player in a game makes the best decision they can, taking into account the decisions of other players, and no player has an incentive to unilaterally change their strategy. In other words, it's a state where each player's strategy is optimal given the strategies chosen by all other players.
- NEAT
  - NEAT, which stands for NeuroEvolution of Augmenting Topologies, is a genetic algorithm-based approach.
- NumPy
  - NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
# O
- optical char recognition
    - TBD
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
- Parameters 
  - They are also known as coefficients or weights, are vital components in linear regression models. They represent the relationships between input variables and the target outcome, such as predicting house prices based on square footage. These parameters are optimized during training to minimize prediction errors, ensuring accurate model performance. Understanding and managing parameters are essential for developing effective linear regression models that generalize well to new data. 
  In linear regression you will have 2 of these per feature.

- phonetic segmentation
    - TBD
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
  - Plotly is a comprehensive, interactive, and web-based data visualization library used for creating rich and customizable graphs, charts, and dashboards. It supports a wide range of chart types including scatter plots, line plots, bar charts, pie charts, heatmaps, 3D plots, and more. Plotly can be integrated with various programming languages such as Python, R, and JavaScript, making it a versatile tool for data visualization across different platforms.
- PyTorch
  - PyTorch is an open-source machine learning framework primarily developed by Facebook's AI Research lab (FAIR). It is widely used for building and training deep learning models due to its flexibility, ease of use, and dynamic computational graph capabilities. PyTorch provides a rich ecosystem of tools and libraries for tasks such as neural network construction, optimization, and deployment.

# Q
- q-learning
    - TBD
# R
- rmsprop 
    - TBD
- radial bases function
  - A Radial Basis Function (RBF) is a mathematical function whose value depends only on the distance from a specified point, known as the center. RBFs are commonly used in various fields including mathematics, engineering, and computer science, particularly in machine learning and numerical analysis. In machine learning, RBFs are often employed as a kernel function in support vector machines (SVMs) for classification tasks or as a basis for interpolation and approximation methods. They are also utilized in neural networks, where they serve as activation functions or as components in architectures such as radial basis function networks (RBFNs).
- RAG, or Retrieval-Augmented Generation
  - It is a model architecture that combines elements of retrieval-based methods and generative models to improve text generation tasks. It was introduced by Facebook AI in 2020.
- Recommendation Systems
  - Recommendation systems are algorithms and techniques used to suggest items or content to users based on their preferences, historical behavior, or similarities with other users. These systems are prevalent in various online platforms such as e-commerce websites, streaming services, social media platforms, and news websites. There are primarily two types of recommendation systems, Collaborative Filtering, Content-based filtering
- Regression 
  - Regression is a supervised learning task where the goal is to predict a continuous numerical value or quantity based on input features. The output variable in regression is continuous and represents a range of possible values.
- regularization
  - Regularization is a technique used in machine learning and statistical modeling to prevent overfitting and improve the generalization performance of models. Overfitting occurs when a model learns to capture noise or irrelevant patterns in the training data, leading to poor performance on unseen data. Regularization introduces additional constraints or penalties on the model parameters during training to discourage complex or extreme solutions that are prone to overfitting.
- relu (rectified linear unit)
  - ReLU, or Rectified Linear Unit, is an activation function commonly used in artificial neural networks, particularly in deep learning models. It is defined as f(x)=max(0,x), which means that the output is zero for negative input values and equal to the input value for positive input values. ReLU activation function is preferred over other activation functions like sigmoid and tanh due to its simplicity, computational efficiency, and ability to alleviate the vanishing gradient problem during training. 

# S
- stocastic effect
    - TBD
- stratego
    - TBD
- style transfer
    - TBD
- Scikit Learn
  - Scikit-learn is an open-source machine learning library for Python that provides simple and efficient tools for data analysis, data preprocessing, machine learning modeling, and evaluation. It is built on top of NumPy, SciPy, and Matplotlib, and it features a consistent and user-friendly interface that facilitates the implementation and experimentation of machine learning algorithms.
- Seaborn
  - Seaborn is a Python data visualization library based on Matplotlib that provides high-level interfaces for creating attractive and informative statistical graphics. It is built on top of Matplotlib and integrates seamlessly with Pandas data structures, making it easy to visualize data from DataFrames and arrays. 
- Support Vector Machines
  - Support Vector Machines (SVM) is a supervised learning algorithm used for classification and regression tasks. SVM is particularly effective in high-dimensional spaces and is widely used for tasks such as text classification, image recognition, and bioinformatics. 
- stochastic gradient descent (sgd)
  - SGD is an iterative optimization algorithm that aims to minimize a cost function by updating the model parameters in the opposite direction of the gradient of the cost function. 
- Sparse Categorical Crossentropy (SCC)
  - Sparse Categorical Crossentropy (SCC) is a loss function used in multiclass classification tasks where the target labels are integers (sparse representation) rather than one-hot encoded vectors. It is commonly used in machine learning frameworks like TensorFlow and Keras.
- Sigmoid function 
  - The sigmoid function is a mathematical function that maps input values to a smooth S-shaped curve, typically ranging from 0 to 1 or -1 to 1. In machine learning, the sigmoid function is commonly used as an activation function in neural networks, particularly in the output layer for binary classification problems. It is also used in logistic regression models to model binary outcomes and in other contexts where the output needs to be bounded between 0 and 1.
- Support vector machines
  - Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks. SVMs are particularly effective in scenarios with high-dimensional data and can handle both linear and non-linear classification problems.
- Spyder
  - A free and open source scientific environment written in Python.
- symbolic regression
  - Symbolic regression is a machine learning technique that aims to find mathematical expressions or formulas that best describe a given dataset. Unlike traditional regression methods that fit parametric models (such as linear regression or polynomial regression) to data, symbolic regression seeks to discover mathematical relationships in an unrestricted, symbolic form.
# T
- tpu?
    - TBD
- transformer network
    - TBD
- tableau
  - Tableau is a powerful data visualization software that allows users to create interactive and shareable dashboards, reports, and visualizations from various data sources. Developed by Tableau Software, now a part of Salesforce, Tableau provides a user-friendly interface for analyzing and presenting data in a visually compelling manner.
- Temporal Difference (TD) Learning 
  - It is a method used in reinforcement learning (RL) for estimating value functions and learning optimal policies by combining ideas from dynamic programming and Monte Carlo methods. TD learning algorithms learn directly from experiences without requiring a model of the environment's dynamics, making them suitable for online and incremental learning.
- Tensor
  - A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array. A vector is a one dimensional or first order tensor and a matrix is a two dimensional or second order tensor.
- tensorflow
  - TensorFlow is an end-to-end open source platform for machine learning.
- TFIDF (Term Frequency-Inverse Document Frequency)
  -  It's a numerical statistic used in information retrieval and text mining to evaluate the importance of a word in a document relative to a collection of documents (corpus). TF-IDF is commonly used as a feature extraction technique in natural language processing (NLP) tasks, such as document classification, clustering, and search engine ranking. (TF): Term frequency measures how frequently a term (word) occurs in a document. (IDF): Inverse document frequency measures the importance of a term across the entire corpus. It's calculated by dividing the total number of documents in the corpus by the number of documents containing the term, and then taking the logarithm of the result. Terms that occur in many documents are given lower weights, while terms that occur in fewer documents are given higher weights.
- Transfer learning 
    - It is a machine learning technique where a model trained on one task is repurposed for another related task. Instead of training a new model from scratch, transfer learning leverages the knowledge gained from solving one problem to help solve a different, but related problem more efficiently.
- trigonometric functions (cos ...)
  - Trigonometric functions are mathematical functions that relate the angles of a right triangle to the ratios of the lengths of its sides. In machine learning, trigonometric functions can be utilized in various ways, especially when dealing with problems involving periodic patterns or cyclic data.
- tree bank
  - A "treebank" refers to a type of corpus or dataset used in natural language processing (NLP) and computational linguistics for syntactic analysis and parsing. It is a collection of sentences annotated with syntactic structures, typically represented as parse trees. Treebanks are essential resources in NLP research and development, as they enable the development and evaluation of syntactic analysis algorithms and models. They are often created manually by linguists or annotated using automated tools and are available for various languages and domains. Well-known treebanks include the Penn Treebank for English, the Universal Dependencies project, and the Prague Dependency Treebank for Czech.
# U
- upsampling 2d
    - TBD
- Underfitting 
  - Underfitting occurs when a machine learning model is too simple to capture the underlying structure of the data. In other words, the model is not able to learn the training data effectively, leading to poor performance on both the training data and new, unseen data. Underfitting typically happens when the model is too simple or lacks the capacity to represent the complexity of the underlying data distribution.
- Univariate Linear Regression
  - Is a statistical method for modeling the relationship between a single predictor variable (independent variable) and a target variable (dependent variable). It involves fitting a linear equation to the data to find the best-fitting line, characterized by its slope (the change in the target variable per unit change in the predictor variable) and intercept (the value of the target variable when the predictor variable is zero). The objective is to minimize the difference between observed and predicted values using techniques like mean squared error (MSE), enabling accurate predictions and inference. Univariate linear regression is foundational in both statistics and machine learning, commonly applied in predictive modeling tasks.
- Upper Confidence Bound (UCB)
  - Upper Confidence Bound (UCB) is a popular algorithm used in the context of multi-armed bandit problems and reinforcement learning to balance exploration and exploitation. The UCB algorithm aims to efficiently allocate resources (e.g., trials, interactions) among multiple actions or arms to maximize cumulative rewards while minimizing regret.
# V
- vanishing or exploding gradient
    - TBD
- vgg16
    - TBD
- Variational Autoencoders (VAEs) 
  - They are a class of generative models in machine learning that combine elements of both autoencoder architectures and probabilistic latent variable models. VAEs are designed to learn low-dimensional representations of input data in an unsupervised manner while simultaneously generating new data samples.
- Vector quantization (VQ) 
  - It is a technique used in signal processing and data compression to reduce the amount of data required to represent a signal or dataset while preserving essential information. It involves partitioning a set of multidimensional data points (vectors) into clusters and representing each vector by the cluster centroid it belongs to.
- VGG16
  - VGG16 is a convolutional neural network (CNN) architecture proposed by the Visual Geometry Group (VGG) at the University of Oxford. It is characterized by its simplicity and uniform architecture, consisting mainly of 3x3 convolutional layers stacked on top of each other, with max-pooling layers interspersed between them. VGG16 is particularly famous for its performance in image classification tasks.
# W
- word2vec
    - TBD
# X
# Y
# Z
