# Disaster Response Pipeline Project

## Introduction

This project is the 2nd project for Udacity's Data Scientist Nanodegree Program. 

In this project, a disaster response dataset was provided by [Figure Eight](https://www.figure-eight.com/). The data set contains two parts - a CSV with disaster messages, and another one with message categories. 

This project aims to build a machine learning model that can take messages and classify them into different categories. The model is then rendered by flask to provide a web GUI - this part is provided by Udacity.

## Folder descriptions

### app

run.py - script to launch the flask web app to render the classification results

### data

process_data.py - script containing ETL pipelines to load, merge, extract, clean and storing data in a sql database

### models

train_classifier.py - script containing ML pipeline to load the cleaned data, and building a ML model (AdaBoostClassifer() in this case), and store the model using pickle


### Instructions:

Install the required dependencies, and follow the steps below 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
