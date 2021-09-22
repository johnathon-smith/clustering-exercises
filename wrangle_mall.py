import numpy as np
import pandas as pd
import env
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#The following function will acquire the mall_customers data from the codeup database
def get_mall_data():
    mall_query = """
        SELECT * FROM customers;
    """

    mall_url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/mall_customers'

    customers = pd.read_sql(mall_query, mall_url)

    return customers

#The following function will summarize the mall data
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

#The following function will plot the individual variable distributions
def get_dists(df):
    for col in df.columns:
        sns.histplot(x = col, data = df)
        plt.title(col)
        plt.show()

#The next two functions will detect outliers
def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_outliers_upper'] = get_upper_outliers(df[col], k)
    return df

#The following function will drop the 'outliers' columns and split the data into train, validate, test sets
def train_validate_test_split(df, seed = 123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''

    #df = df.drop(columns = ['customer_id_outliers_upper', 'age_outliers_upper', 'annual_income_outliers_upper', 'spending_score_outliers_upper'])

    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

#The following function will create dummy variables, remove the original column, and return the new data frames
def get_dummy_vars(train, validate, test, cols_to_encode):
    train_dummies = pd.get_dummies(train[cols_to_encode], dummy_na=False, drop_first=True)
    train = pd.concat([train, train_dummies], axis = 1).drop(columns = cols_to_encode)

    validate_dummies = pd.get_dummies(validate[cols_to_encode], dummy_na=False, drop_first=True)
    validate = pd.concat([validate, validate_dummies], axis = 1).drop(columns = cols_to_encode)

    test_dummies = pd.get_dummies(test[cols_to_encode], dummy_na=False, drop_first=True)
    test = pd.concat([test, test_dummies], axis = 1).drop(columns = cols_to_encode)

    return train, validate, test

#The following function drops entries and columns based on the percentage of missing values
def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df

#The following function will scale the data and return the scaled dfs
def scale_data(X_train, X_validate, X_test):
    #Create the scaler
    scaler = StandardScaler()

    #Fit the scaler on X_train
    scaler.fit(X_train)

    #Transform the data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validate_scaled, X_test_scaled

#The following function will allow the comparison of the original distributions with the new scaled distributions
def compare_dists(train, cols_to_scale, cols_scaled):

    plt.figure(figsize=(18,6))

    for i, col in enumerate(cols_to_scale):
        i += 1
        plt.subplot(2,4,i)
        train[col].plot.hist()
        plt.title(col)

    for i, col in enumerate(cols_scaled):
        i += 5
        plt.subplot(2,4,i)
        train[col].plot.hist()
        plt.title(col)

    plt.tight_layout()
    plt.show()