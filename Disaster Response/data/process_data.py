'''
Project Name: Disaster response pipeline - part of Udacity's Nanodegree Program

This module can be used to load, extract, and transform data to prepare for the next module

Args: message_filepath = file location of the CSV file containing messages
      categories_file_path = file location of the CSV file containing message categories
      database_filepath = file location of the sql database storage file
'''

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    input - file path to the message and category file
    output - combined DataFrame containing both messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def convert(text):
    '''
    input - category text strings 
    output - category values 0 or 1
    '''

    cat = int(text[-1])
    cat = np.ceil(cat / 2)
    return int(cat)



def clean_data(df):
    '''
    input - dataframe from load_data
    output - dataframe with category splitted in to columns and values filled
    '''
    # Splitting the categories string
    categories = df['categories'].str.split(pat=';', expand=True)

    # Renaming categories column names
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [s[:-2] for s in row]
    categories.columns = category_colnames

    # Setting category value to numbers
    for column in categories:
        categories[column] = categories[column].apply(convert)
    
    # replacing df categories columns with new category columns
    df.drop(labels=['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Removing duplicates

    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''
    Saving df to database file 
    input: df - processed dataframe
           database_filename - database file name and location
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_message', engine, index=False, if_exists='replace')

      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()