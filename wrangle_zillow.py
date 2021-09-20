import numpy as np
import pandas as pd
import env

#The following function will acquire the zillow data
def get_zillow_data():
    #Write the SQL query to find the right data
    zillow_query = """
    SELECT * FROM properties_2017
    LEFT JOIN predictions_2017 ON predictions_2017.parcelid = properties_2017.parcelid
    LEFT JOIN airconditioningtype ON airconditioningtype.airconditioningtypeid = properties_2017.airconditioningtypeid
    LEFT JOIN architecturalstyletype ON architecturalstyletype.architecturalstyletypeid = properties_2017.architecturalstyletypeid
    LEFT JOIN buildingclasstype ON buildingclasstype.buildingclasstypeid = properties_2017.buildingclasstypeid
    LEFT JOIN storytype ON storytype.storytypeid = properties_2017.storytypeid
    LEFT JOIN typeconstructiontype ON typeconstructiontype.typeconstructiontypeid = properties_2017.typeconstructiontypeid
    LEFT JOIN heatingorsystemtype ON heatingorsystemtype.heatingorsystemtypeid = properties_2017.heatingorsystemtypeid
    LEFT JOIN propertylandusetype ON propertylandusetype.propertylandusetypeid = properties_2017.propertylandusetypeid
    WHERE (predictions_2017.transactiondate >= '2017-01-01'
        AND predictions_2017.transactiondate <= '2017-12-31')
        AND properties_2017.latitude IS NOT NULL
        AND properties_2017.longitude IS NOT NULL;
    """

    #Write the url for the zillow database
    zillow_url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'

    #Load the data from the Codeup database
    zillow = pd.read_sql(zillow_query, zillow_url)

    #Duplicate parcelids still need to be dropped.
    #Place df in ascending order of transaction date then drop_duplicatese with keep = last
    zillow.transactiondate.sort_values()

    #Now drop duplicates, but keep last occurence. This will ensure the latest transaction date is kept
    zillow = zillow.drop_duplicates(subset = 'parcelid', keep = 'last')

    return zillow

#The following function returns a dataframe where each row is an atttribute name, the first column is the number of rows with missing values 
#for that attribute, and the second column is percent of total rows that have missing values for that attribute.
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

#The following function calculates the number of columns missing, percent of columns missing, and number of rows with n columns missing.
def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing

#The following function will summarize the zillow data. It will include null value counts and percentages.
def summarize(df):
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3).to_markdown())
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe().to_markdown())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')

#The following function drops entries and columns based on the percentage of missing values
def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df

#The following function will prepare the zillow data by removing or imputing null values, reducing the property types to single units,
#and dropping unnecessary columns
def prepare_zillow(zillow):
    #Drop duplicate columns because they are giving me issues
    zillow = zillow.T.drop_duplicates(keep = 'last').T

    #The following propertylandusetypeid's are single unit properties: 
    #261, 262, 263, 264, 265, 266
    zillow = zillow[(zillow.propertylandusetypeid >= 261) & (zillow.propertylandusetypeid <= 266)]

    #Remove entries and columns with too many missing values
    zillow = handle_missing_values(zillow)

    #Fill missing 'home_area' values with 'home_area' median
    zillow.calculatedfinishedsquarefeet = zillow.calculatedfinishedsquarefeet.fillna(zillow.calculatedfinishedsquarefeet.median())

    #Impute the lotsizesquarefeet median value
    lot_size_median = zillow.lotsizesquarefeet.median()
    zillow.lotsizesquarefeet = zillow.lotsizesquarefeet.fillna(lot_size_median)

    #Impute heatingorsystemdesc with the most common value for this column, 'Central'
    zillow.heatingorsystemdesc = zillow.heatingorsystemdesc.fillna('Central')

    #Since there is no description to go with buildingqualitytypeid, drop the column
    zillow.drop(columns = ['buildingqualitytypeid', 'heatingorsystemtypeid', 'propertyzoningdesc', 'unitcnt', 'id', 'propertylandusetypeid'], inplace = True)

    #Drop all other missing entries
    zillow.dropna(inplace = True)

    return zillow
