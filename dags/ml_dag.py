import pandas as pd
import warnings
import numpy as np
import boto3
import datetime
from datetime import datetime, timedelta
import airflow.models
from pyathena import connect
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

aws_access_key_id_source = airflow.models.Variable.get("aws_access_key_id")
aws_secret_access_key_source = airflow.models.Variable.get("aws_secret_access_key")
aws_staging_dir = airflow.models.Variable.get("aws_staging_dir")

# EXTRACTION - TASK ONE
def extraction():
    def connect_aws():
        conn = connect(aws_access_key_id=aws_access_key_id_source,
                    aws_secret_access_key=aws_access_key_id_source,
                    s3_staging_dir=aws_staging_dir,
                    region_name='sa-east-1')
        return conn
    
    data = pd.read_sql("SELECT * FROM raw.fiec_industry_data", connect_aws())
    return data


# TRANSFORMATION - TASK TWO
def transformation(**kwargs):
    ti = kwargs['ti']  # Get the TaskInstance object
    data = ti.xcom_pull(task_ids='extract_data') 
    data.columns = [col.title() for col in data.columns]

    # Categorical column for Profit
    data['Profit_Class'] = (data['Profit'] > 0).astype(int)
    print(data.columns)
    # Filter columns of type 'object'
    object_cols = data.select_dtypes(include=['object']).columns

    object_cols_less_than_100 = [col for col in object_cols if col not in ['Product_Name', 'Customer_Name']]
    object_cols_higher_than_100 = ['Customer_Name','Product_Name']

    # Calculate the counts of each customer
    customer_counts = data['Customer_Name'].value_counts()

    # Calculate the average and maximum counts
    average_count = customer_counts.mean()
    max_count = customer_counts.max()
    print('Average: ', average_count)
    print('Max: ', max_count)
    # Create a dictionary to map the encoding
    encoding_dict = {}

    for customer, count in customer_counts.items():
        if count == 1:
            encoding_dict[customer] = 1
        elif 1 < count < average_count:
            encoding_dict[customer] = 2
        elif average_count <= count < max_count:
            encoding_dict[customer] = 3
        elif count == max_count:
            encoding_dict[customer] = 4

    # Map the encoding to the Customer_Name column
    data['Encoded_Customer'] = data['Customer_Name'].map(encoding_dict)


    # Product Name goes from 1 to 5

    # Calculate the counts of each product
    product_counts = data['Product_Name'].value_counts()

    # Create a dictionary to map the encoding
    encoding_dict = {}

    for product, count in product_counts.items():
        if count == 1:
            encoding_dict[product] = 1
        else:
            encoding_dict[product] = 2

    # Map the encoding to the Product_Name column
    data['Encoded_Product'] = data['Product_Name'].map(encoding_dict)

    data_preprocessed = pd.get_dummies(data, columns = object_cols_less_than_100)

    data_preprocessed = data_preprocessed.set_index('Order_Id').drop(columns = ['Customer_Name', 'Product_Name'])

    # Separate data into feature and target

    return data_preprocessed

def training(**kwargs):
# TRAINING - TASK THREE
    ti = kwargs['ti']  # Get the TaskInstance object
    data_preprocessed = ti.xcom_pull(task_ids='transform_data') 

    X = data_preprocessed.drop(columns = 'Profit')
    y = data_preprocessed.reset_index()['Profit']

    # Remove High Correlated Columns
    def correlated_columns(df, threshold=0.80):
        return (
            df.corr()
            .pipe(
                lambda df1: pd.DataFrame(
                    np.tril(df1, k=-1),
                    columns=df.columns,
                    index=df.columns,
                )
            )
        .stack()
        .rename("pearson")
        .pipe(
            lambda s: s[
                s.abs() > threshold
            ].reset_index()
        )
            .query("level_0 not in level_1")
        )

    cr = correlated_columns(X)
    X = X.drop(columns = cr['level_0'])
    #print('Removed columns with correlation higher than 0.8:', cr['level_0'])
    # Divide data into train and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 1)

    # Creating SelectKBest object with f_regression scoring function
    selector = SelectKBest(score_func=f_regression, k=20)  # Select the top 20 features

    # Fitting the selector to the training data
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Getting the indices of the selected features
    selected_indices = selector.get_support(indices=True)

    # Print the selected feature indices
    print("Selected Feature Indices:", selected_indices)


    # Filter the X_train and X_test
    X_train_filtered = X_train.iloc[:,selected_indices]
    X_test_filtered = X_test.iloc[:,selected_indices]

    return [X_train_filtered,X_test_filtered,y_train.to_frame(),y_test.to_frame()]

    # PREDICTION - TASK FOUR
def prediction(**kwargs):
    ti = kwargs['ti']  # Get the TaskInstance object
    X_train_filtered = ti.xcom_pull(task_ids='train_model')[0]
    X_test_filtered = ti.xcom_pull(task_ids='train_model')[1]
    y_train = ti.xcom_pull(task_ids='train_model')[2]
    y_test = ti.xcom_pull(task_ids='train_model')[3]
    # Creating and fitting the SVR model with a linear kernel
    print(X_test_filtered)
    svr_linear = SVR(kernel='linear')
    svr_linear.fit(X_train_filtered, y_train)
    # Making predictions
    y_pred = svr_linear.predict(X_test_filtered)
    mse = mean_squared_error(y_test, y_pred)
    # Calculating Mean Squared Error
    print("Mean Squared Error:", mse)

    # Convert y_pred and y_test to DataFrame for printing
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted'])
    y_test_df = pd.DataFrame(y_test, columns=['Actual'])
    
    return [y_pred_df, y_test_df]

# EVALUATION - TASK FIVE
def evaluation(**kwargs):
    ti = kwargs['ti']  # Get the TaskInstance object
    y_test = ti.xcom_pull(task_ids='predict')[0].iloc[:,0].values
    y_pred = ti.xcom_pull(task_ids='predict')[1].iloc[:,0].values
    print(y_pred)
    # Calculate the percentage difference between y_pred and y_test
    percentage_difference = np.abs((y_pred - y_test) / y_test) * 100

    # Count the number of satisfactory predictions (where percentage difference is less than 20%)
    satisfactory_predictions = np.sum(percentage_difference < 20)

    # Calculate the percentage of satisfactory predictions
    percentage_satisfactory = (satisfactory_predictions / len(y_test)) * 100
    
    print("Percentage of predictions with less than 20% difference from actual value:", percentage_satisfactory, "%")
    return  percentage_satisfactory

# Definição dos argumentos padrão da DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 13),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definição da DAG
with airflow.DAG(
    'model_pipeline',
    default_args=default_args,
    description='Pipeline for model extraction, transformation, training, prediction, and evaluation',
    schedule_interval=None,
) as dag:

    # Definição das tasks
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extraction,
        dag=dag,
    )

    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transformation,
        dag=dag,
        provide_context=True
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=training,
        dag=dag,
        provide_context=True
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=prediction,
        dag=dag,
        provide_context=True
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluation,
        dag=dag,
        provide_context=True
    )

# Definição das dependências entre as tasks
extract_task >> transform_task >> train_task >> predict_task >> evaluate_task
