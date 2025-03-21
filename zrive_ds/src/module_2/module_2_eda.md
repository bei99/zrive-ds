### 0. Get the Data

#### 0.1 Pass AWS Credentials and inspect the content of the bucket


```python
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(dotenv_path='/Users/Bei/Documents/Bei/Zrive/zrive-ds/zrive_ds/src/module_2/passwords.env')

# Get the AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


client = boto3.client('s3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

bucket_name = 'zrive-ds-data'

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)
bucket_objects = []
for obj in bucket.objects.all():
    print(obj.key)
    bucket_objects.append(obj.key)
```

    groceries/box_builder_dataset/feature_frame.csv
    groceries/sampled-datasets/abandoned_carts.parquet
    groceries/sampled-datasets/inventory.parquet
    groceries/sampled-datasets/orders.parquet
    groceries/sampled-datasets/regulars.parquet
    groceries/sampled-datasets/users.parquet
    groceries/trained-models/model.joblib


##### 0.2 Download the sampled datasets


```python
parquet_files = [i for i in bucket_objects if i.endswith(".parquet")]

destination_dir = 'groceries_datasets'
os.makedirs(destination_dir, exist_ok=True)

for s3_path in parquet_files:
    # Get just the filename from the S3 path
    filename = s3_path.split('/')[-1]
    destination_file = os.path.join(destination_dir, filename)
    
    try:
        client.download_file(bucket_name, s3_path, destination_file)
        print(f"Downloaded '{s3_path}' from bucket '{bucket_name}' to '{destination_file}'")
    except ClientError as e:
        print(f"Error downloading '{s3_path}' from bucket '{bucket_name}': {e}")
```

    Downloaded 'groceries/sampled-datasets/abandoned_carts.parquet' from bucket 'zrive-ds-data' to 'groceries_datasets/abandoned_carts.parquet'
    Downloaded 'groceries/sampled-datasets/inventory.parquet' from bucket 'zrive-ds-data' to 'groceries_datasets/inventory.parquet'
    Downloaded 'groceries/sampled-datasets/orders.parquet' from bucket 'zrive-ds-data' to 'groceries_datasets/orders.parquet'
    Downloaded 'groceries/sampled-datasets/regulars.parquet' from bucket 'zrive-ds-data' to 'groceries_datasets/regulars.parquet'
    Downloaded 'groceries/sampled-datasets/users.parquet' from bucket 'zrive-ds-data' to 'groceries_datasets/users.parquet'


### 1. Understanding the problem space

#### 1.1 Inspect Tables

##### 1.1.1 Inspect the Orders Table


```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

file_path = "groceries_datasets/orders.parquet"


# Read the downloaded Parquet file into a pandas DataFrame
orders = pd.read_parquet(file_path)

# Display the first few rows
display(orders.head())
orders.info()
orders.dtypes
orders.count()
orders.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.773000e+03</td>
      <td>8773</td>
      <td>8773</td>
      <td>8773.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.684684e+12</td>
      <td>2021-08-22 03:54:18.750028</td>
      <td>2021-08-21 12:47:21.262966</td>
      <td>2.445116</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.204073e+12</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30 00:00:00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.690255e+12</td>
      <td>2021-04-25 11:50:37</td>
      <td>2021-04-25 00:00:00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.846692e+12</td>
      <td>2021-10-11 11:29:44</td>
      <td>2021-10-11 00:00:00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.945086e+12</td>
      <td>2022-01-03 18:14:23</td>
      <td>2022-01-03 00:00:00</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.026732e+12</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14 00:00:00</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.145437e+11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.707693</td>
    </tr>
  </tbody>
</table>
</div>



##### 1.1.2 Inspect the Abandoned Carts Table


```python
# Specify the path to your parquet file
file_path = "groceries_datasets/abandoned_carts.parquet"


# Read the downloaded Parquet file into a pandas DataFrame
abandoned_carts = pd.read_parquet(file_path)

# Display the first few rows
display(abandoned_carts.head())
abandoned_carts.info()
abandoned_carts.dtypes
abandoned_carts.count()
abandoned_carts.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 5457 entries, 0 to 70050
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   id          5457 non-null   int64         
     1   user_id     5457 non-null   object        
     2   created_at  5457 non-null   datetime64[us]
     3   variant_id  5457 non-null   object        
    dtypes: datetime64[us](1), int64(1), object(2)
    memory usage: 213.2+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.457000e+03</td>
      <td>5457</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.161881e+13</td>
      <td>2021-12-20 11:07:10.198460</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.285856e+13</td>
      <td>2020-05-20 13:53:24</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.133401e+13</td>
      <td>2021-11-13 19:52:17</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.167062e+13</td>
      <td>2021-12-27 13:14:57</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.192303e+13</td>
      <td>2022-01-30 08:35:19</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.223385e+13</td>
      <td>2022-03-13 14:12:10</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.028679e+11</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



##### 1.1.3 Inspect the Inventory Table


```python
# Specify the path to your parquet file
file_path = "groceries_datasets/inventory.parquet"


# Read the downloaded Parquet file into a pandas DataFrame
inventory = pd.read_parquet(file_path)

# Display the first few rows
display(inventory.head())
inventory.info()
inventory.dtypes
inventory.count()
inventory.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.733000e+03</td>
      <td>1733.000000</td>
      <td>1733.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.694880e+13</td>
      <td>6.307351</td>
      <td>7.028881</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.725674e+12</td>
      <td>7.107218</td>
      <td>7.660542</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.361529e+13</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.427657e+13</td>
      <td>2.490000</td>
      <td>2.850000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.927260e+13</td>
      <td>3.990000</td>
      <td>4.490000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.948318e+13</td>
      <td>7.490000</td>
      <td>8.210000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.016793e+13</td>
      <td>59.990000</td>
      <td>60.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### 1.1.4 Inspect the Regulars Table


```python
# Specify the path to your parquet file
file_path = "groceries_datasets/regulars.parquet"


# Read the downloaded Parquet file into a pandas DataFrame
regulars = pd.read_parquet(file_path)

# Display the first few rows
display(regulars.head())
regulars.info()
regulars.dtypes
regulars.count()
regulars.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[us]
    dtypes: datetime64[us](1), int64(1), object(1)
    memory usage: 565.8+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.810500e+04</td>
      <td>18105</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.511989e+13</td>
      <td>2021-08-15 02:27:30.703728</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.361527e+13</td>
      <td>2020-04-30 13:09:27</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.382643e+13</td>
      <td>2021-03-21 10:41:42</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.422171e+13</td>
      <td>2021-10-16 09:11:26</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.448855e+13</td>
      <td>2022-01-14 22:35:14</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.013362e+13</td>
      <td>2022-03-14 07:49:24</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.171237e+12</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



##### 1.1.5 Inspect the Users Table


```python
# Specify the path to your parquet file
file_path = "groceries_datasets/users.parquet"


# Read the downloaded Parquet file into a pandas DataFrame
users = pd.read_parquet(file_path)

# Display the first few rows
display(users.head())
users.info()
users.dtypes
users.count()
users.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c35...</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.787692</td>
      <td>2.003077</td>
      <td>0.707692</td>
      <td>0.076923</td>
      <td>0.636923</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.365753</td>
      <td>0.869577</td>
      <td>1.026246</td>
      <td>0.289086</td>
      <td>0.995603</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### Groceries Dataset Schema:

To investigate/validate:
- Users table:
    - Investigate:
        - What does the poeple/adults/children/pets/babies mean, and how it works? why do they have the same number of nulls?
        - Column customer_cohort_month, what does it mean?
        - column user_nuts1, what does it mean? is it a categorical variable?
        - Column user_segment, what does it mean? is it a categorical variable?
    - Validate:
        - Are users only those who purchased an item or anyone who signed up?
            - Test: if there are users that do not have any purchase (first_ordered_at has NULLs) 
- Orders table:
    - Validate:
        - It looks like ordered_items is a list with the variant_id of the products purchased
            - Check that the items in ordered_items exists as variant_id

---

To do:
- In the case that ordered_items in the orders table and variant_id in abandoned_carts table are the product_ids, it would be easier to unnest those items so there is only one product per row.

- user_id is an object, it would be easier to manipulate if we converted it to a string.
    
- Investigate and validate hypotheses.
        
        
        
        
        
        
      

##### Users Table

What does the variable count_people mean?


```python
users.head()


user_categories = users[['count_people','count_adults','count_children','count_babies','count_pets']]

user_categories_clean = user_categories.dropna(how='all')

try:
    user_categories_clean['count_people'].sum() == user_categories_clean['count_adults'].sum() + user_categories_clean['count_children'].sum() + user_categories_clean['count_babies'].sum()
    print('The variable count_people is the sum of adults, children and babies')
except:
    print("Don't know what count_people means")
```

    The variable count_people is the sum of adults, children and babies



```python
user_categories_clean.groupby(list(user_categories_clean.columns)).size().reset_index(name="count")

df_melted = user_categories_clean.melt(var_name='Variable', value_name='Value')

sns.displot(df_melted,x='Value',hue='Variable',kind='kde')

print(df_melted.pivot_table(index='Value', columns='Variable', aggfunc='size', fill_value=0))
```

    Variable  count_adults  count_babies  count_children  count_people  count_pets
    Value                                                                         
    0.0                  3           302             195             2         193
    1.0                 71            21              55            57          87
    2.0                201             2              58            97          28
    3.0                 32             0              12            68           8
    4.0                 11             0               3            67           6
    5.0                  5             0               1            24           2
    6.0                  1             0               1             8           1
    7.0                  1             0               0             1           0
    8.0                  0             0               0             1           0



    
![png](module_2_eda_files/module_2_eda_20_1.png)
    



```python
users['has_personal_attributes'] = np.where(users['count_people'].isnull(),False,True)
users['year_month'] = users['first_ordered_at_trunc'].dt.to_period('M')

#  Extract year-month

monthly_counts = users.groupby(['year_month', 'has_personal_attributes']).size().unstack(fill_value=0)

# Convert to percentage (normalize)
monthly_percentage = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100

# Plot
monthly_percentage.plot(kind='bar', stacked=True, color=['salmon', 'skyblue'], figsize=(10, 5))

# Customize plot
plt.title('% Users with Personal attributes by first purchased month')
plt.ylabel('%')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.legend(['False', 'True'], title="Has user personal attributes")
plt.show()

```


    
![png](module_2_eda_files/module_2_eda_21_0.png)
    


What does customer_cohort_month?


```python
#truncate first order date to month
users['first_ordered_at_trunc'] = pd.to_datetime(users['first_ordered_at']).dt.to_period("M").dt.to_timestamp()

users['customer_cohort_month'] = pd.to_datetime(users['customer_cohort_month']).dt.to_period("M").dt.to_timestamp()


users["is_customer_cohort_first_order"] = users['first_ordered_at_trunc'] == users['customer_cohort_month']


print(users['is_customer_cohort_first_order'].value_counts())


print(users[["first_ordered_at","first_ordered_at_trunc","customer_cohort_month"]][users["is_customer_cohort_first_order"] == False])


```

    is_customer_cohort_first_order
    True     4982
    False       1
    Name: count, dtype: int64
             first_ordered_at first_ordered_at_trunc customer_cohort_month
    3699  2020-10-01 00:37:56             2020-10-01            2020-09-01


The column customer_cohort_month is a user cohort truncated by the month in which the user made the first purchase. There is an exception in one record, which might indicate the the timezone on the first_ordered_at column might be different than what it was used for the customer_cohort_month


```python
print(users['user_nuts1'].value_counts())

```

    user_nuts1
    UKI    1318
    UKJ     745
    UKK     602
    UKH     414
    UKD     358
    UKM     315
    UKE     303
    UKG     295
    UKF     252
    UKL     224
    UKC     102
    UKN       4
    Name: count, dtype: int64


The variable user_nuts1 is a categorical variable that indicates a geographical zone from the UK.


```python
# get the number of items ordered within the same order
orders['ordered_items_sum'] = orders['ordered_items'].apply(len)

# get the number of times a user ordered in total
user_orders = orders.groupby("user_id")["id"].count().reset_index()
user_orders.columns = ["user_id", "order_count"] 

# get the number of items a user ordered in total
user_items_sum = orders.groupby("user_id")["ordered_items_sum"].sum().reset_index()
user_items_sum.columns = ["user_id", "total_items_ordered"]

# join the orders and number of items
user_orders_and_items = pd.merge( user_orders,user_items_sum, how='left', on='user_id')

# join the user segment with the orders data
user_segments_w_orders = pd.merge(users[['user_id','user_segment']],user_orders_and_items,how='left',on='user_id')


analysis = user_segments_w_orders.groupby("user_segment").agg({
    'order_count': ['count', 'mean', 'min', 'max', 'sum'],
    'total_items_ordered': ['mean', 'min', 'max', 'sum']
})

analysis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">order_count</th>
      <th colspan="4" halign="left">total_items_ordered</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>sum</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>user_segment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Proposition</th>
      <td>2340</td>
      <td>1.785043</td>
      <td>1</td>
      <td>22</td>
      <td>4177</td>
      <td>27.937179</td>
      <td>7</td>
      <td>499</td>
      <td>65373</td>
    </tr>
    <tr>
      <th>Top Up</th>
      <td>2643</td>
      <td>1.738933</td>
      <td>1</td>
      <td>25</td>
      <td>4596</td>
      <td>16.112372</td>
      <td>1</td>
      <td>389</td>
      <td>42585</td>
    </tr>
  </tbody>
</table>
</div>



It doesn't look like there is any correlation between the number of orders or the number of items ordered for the user_segment variable. A possible explanation for the segment meaning is in the products they purchased.


```python
added_to_regulars = regulars.groupby('user_id')['created_at'].min()

users_w_regulars = pd.merge(user_segments_w_orders,added_to_regulars,how='left',on='user_id')


result = users_w_regulars.groupby('user_segment')['created_at'].agg([
    ('added_products_to_regulars', lambda x: x.isna().sum()),
    ('not_added_products_to_regulars', lambda x: x.notna().sum())
])

print(result)
```

                  added_products_to_regulars  not_added_products_to_regulars
    user_segment                                                            
    Proposition                         1556                             784
    Top Up                              1979                             664


It does not look like there is a correlation between the user_segment and the user having added an item to their regulars


```python
if users['first_ordered_at'].isna().sum()==0:
    print("All users are paying customers") 
else:
    print("There are users that just registered")
```

    All users are paying customers


#### Orders


```python
# get only the necessary columns
ordered_items = orders[['order_date','ordered_items']]

# unnest the list of items
ordered_items= ordered_items.explode('ordered_items')


# get the first, last and number of purchases for a given item
orders_agg = ordered_items.groupby('ordered_items').agg({
    'order_date':['min','max','count']
}).reset_index()

# rename the columns
orders_agg.columns =['variant_id','first_ordered','last_ordered','no_times_ordered']

ordered_products = pd.merge(orders_agg,inventory, how='left',on='variant_id')

ordered_products['has_price'] = np.where(ordered_products['price'].isnull(),False,True)

ordered_products.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>first_ordered</th>
      <th>last_ordered</th>
      <th>no_times_ordered</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
      <th>has_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33615294398596</td>
      <td>2020-10-24</td>
      <td>2022-03-10</td>
      <td>88</td>
      <td>2.99</td>
      <td>3.0</td>
      <td>hollings</td>
      <td>dog-food</td>
      <td>[dog-treats]</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33618830196868</td>
      <td>2020-05-01</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33618835243140</td>
      <td>2020-04-30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33618835964036</td>
      <td>2020-04-30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33618846580868</td>
      <td>2020-05-01</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python

#  Extract year-month
ordered_products['year_month'] = ordered_products['first_ordered'].dt.to_period('M')

monthly_counts = ordered_products.groupby(['year_month', 'has_price']).size().unstack(fill_value=0)

# Convert to percentage (normalize)
monthly_percentage = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100

# Plot
monthly_percentage.plot(kind='bar', stacked=True, color=['salmon', 'skyblue'], figsize=(10, 5))

# Customize plot
plt.title('% Items sold that are in inventary (by first sold at)')
plt.ylabel('%')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.legend(['False', 'True'], title="Is product in inventory")
plt.show()
```


    
![png](module_2_eda_files/module_2_eda_34_0.png)
    


We do not have inventory details for some products. For products sold before 2021, the % of products where we do not have invontry data is around 50%, from 2021 onwards that drops dramatically to around 20%.

What we know:

- We have a dataset from an ecommerce site with the following schema:

![alt text](groceries_schema.png "Groceries Schema")


---

We know that users have not just signed up but they must have purchased a product to qualify as users. 

We have segmented them:
- by when they made their first purchase (Monthly cohort).
- by their household characteristics: number of people and type (individual, couple, childs/pets). This data might not be reliable because most of this data is from 04/2020 - 08/2020 cohorts.
- UK geographical zone.
- User segment (Top Up/ Proposition), we do not know how this segmentation is done. We know it is not based on the number of orders, the number of items per order, or adding items to regulars.


Regarding orders, we have missing data for products description and pricing that started being sold in 2020. 

