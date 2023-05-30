# Spam Classification Project

This project focuses on email classification using different machine learning models from the scikit-learn library. The goal is to develop a model that can accurately classify emails into different ham and spam.

Additionally, the project incorporates natural language processing (NLP) techniques using the nltk library. Tasks such as stemming, lemmatization, tokenization, and stop-word removal are performed to preprocess the text data before training the models.

## Classification models

The classification models employed in this project are as follows:

- K-Nearest Neighbors (KNN) Classifier.

- Logistic Regression.

- Naive Bayes Classifiers.

- Support Vector Machines (SVM).

- Decision Tree Classifier.

- Multilayer Perceptron (MLP) Classifier.

## Evaluation

Evaluation metrics such as accuracy, precision, recall, and F1 score are used to assess the performance of each model. Confusion matrices and ROC curves are also generated for further analysis.

## Hyper parameters tuning

Hyperparameter tuning is performed using RandomizedSearchCV and GridSearchCV from scikit-learn. Randomized search randomly samples a subset of hyperparameters, while grid search exhaustively evaluates all possible combinations of hyperparameters. These techniques help optimize the model's performance by finding the best set of hyperparameters.


