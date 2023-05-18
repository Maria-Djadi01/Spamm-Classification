# Spam Classification Project

This project focuses on email classification using different machine learning models from the scikit-learn library. The goal is to develop a model that can accurately classify emails into different categories.

The project begins by importing the necessary libraries, including numpy, pandas, matplotlib, and re for data manipulation and visualization. The scikit-learn library is used for various tasks such as data splitting, cross-validation, hyperparameter tuning, and implementing different classification models.

## Classification models

The classification models employed in this project are as follows:

- K-Nearest Neighbors (KNN) Classifier: This model classifies instances based on the similarity of their features to the k nearest neighbors in the training data.

- Logistic Regression: It is a linear model used for classification by estimating the probability of the input belonging to a certain class.

- Naive Bayes Classifiers: This project utilizes three different Naive Bayes classifiers: GaussianNB, MultinomialNB, and BernoulliNB. These classifiers are based on the Bayes theorem and assume independence between features.

- Support Vector Machines (SVM): SVM is a powerful algorithm for classification that finds the optimal hyperplane to separate different classes.

- Decision Tree Classifier: This model constructs a decision tree by recursively partitioning the training data based on different features.

- Multilayer Perceptron (MLP) Classifier: An artificial neural network model with multiple layers of nodes. It is trained using backpropagation and can handle complex classification tasks.

## Evaluation

Evaluation metrics such as accuracy, precision, recall, and F1 score are used to assess the performance of each model. Confusion matrices and ROC curves are also generated for further analysis.

## Hyper parameters tuning

Hyperparameter tuning is performed using RandomizedSearchCV and GridSearchCV from scikit-learn. Randomized search randomly samples a subset of hyperparameters, while grid search exhaustively evaluates all possible combinations of hyperparameters. These techniques help optimize the model's performance by finding the best set of hyperparameters.

Additionally, the project incorporates natural language processing (NLP) techniques using the nltk library. Tasks such as stemming, lemmatization, tokenization, and stop-word removal are performed to preprocess the text data before training the models.
