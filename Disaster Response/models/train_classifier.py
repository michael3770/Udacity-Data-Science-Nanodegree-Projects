'''
Project Name: Disaster response pipeline - part of Udacity's Nanodegree Program

This module load dataset from the database we saved in the last step and train the model based on our data, and save the model using pickle 

'''

# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(database_filepath):
    '''
    Load database into a dataframe for processing
    input - database_filepath
    output - X -> dataframe with features
             y -> dataframe with labels
             category_names -> list of category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_message', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    return X, y, y.columns



def tokenize(text):
    '''
    This function will convert messages into tokenized words so the dataset can be processed by the following machine learning pipeline
    The messages are mostly english sentences and this function will do tokenization, convert to lower case, lemmatization, and stemming
    input - message(string)
    output - list of processed key words(list of strings)
    '''
    msg = text.lower()
    
    # matching words and numbers use regex
    msg = re.sub(r"[^A-Za-z0-9]", " ", msg)

    # tokentization, remove stop words, stem and lemmatization
    words = word_tokenize(msg)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return words



def build_model():
    '''
    Classifier model function. Based on the evaluation results from ML pipeline prep notebook, AdaBoostClassifier will be used here
    '''
    pipeline = Pipeline([
    ('vert', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
      'clf__estimator__algorithm': ['SAMME', 'SAMME.R'],
      'clf__estimator__n_estimators': [25, 50, 75]
    }
    # use a custom scorer with f1_score
    score = make_scorer(multi_f1)
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=score, verbose=10)
    
    return cv

# for some reason classification_report does not work in multioutput multilabel, use the function below to get metrics

def get_scores(y_true, y_pred):
    '''
    input :vector from dataset (y_train, y_test)
           predicted vector from model (y_train_pred, y_test_pred)
    output: Precision, Recall and F1 score in a dict
    '''
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = 2 * precision * recall / (precision + recall)
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'f1_score': f1}

def get_score_df(y_true, y_pred):
    '''
    input: Trainig/test matrix,
           predicted matrix
    output: metrics dataframe
    '''
    score = []
    for i in range(len(y_true.columns)):
        current_score = get_scores(y_true.iloc[:, i], y_pred[:, i])
        score.append(current_score)
    score_df = pd.DataFrame(score)
    score_df.index = y_true.columns
    return score_df


def multi_f1(y_true, y_pred):
    '''
    custom scoring matrics based on f1_score
    input - true matrix, prediction matrix
    output - average f1_score 
    '''
    score_df = get_score_df(y_true, y_pred)
    return np.mean(score_df['f1_score'])

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate ML model performance metrics
    input-
    model - trained model
    X_test - test set messages
    Y_test - test set labels
    category_names: Each category name

    output:
    printed metrics for each message category

    '''
    Y_test_pred = model.predict(X_test)

    df = get_score_df(Y_test, Y_test_pred)
    
    for column in category_names:
        print(df.loc[column, :])



def save_model(model, model_filepath):
    '''
    Saving machine learning model
    input:
    model - machine learning model
    model_filepath - filepath for pickle saved module
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()