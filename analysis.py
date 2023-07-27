from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, count, to_timestamp, monotonically_increasing_id, desc, when, sum as _sum, monotonically_increasing_id
from pyspark.sql.functions import dayofmonth, weekofyear, month, year
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import StandardScaler, VectorAssembler

import os 

import numpy as np

import matplotlib.pyplot as plt
from math import isnan


def get_schema():
    schema = StructType([])

    schema.add('timestamp', StringType(), True)
    schema.add('from_bank', IntegerType(), True)
    schema.add('from_account', StringType(), True)
    schema.add('to_bank', IntegerType(), True)
    schema.add('to_account', StringType(), True)

    schema.add('amount_received', DoubleType(), True) 
    schema.add('receiving_currency', StringType(), True)
    schema.add('amount_paid', DoubleType(), True)
    schema.add('payment_currency', StringType(), True)
    schema.add('payment_format', StringType(), True)
    schema.add('is_laundering', IntegerType(), True)

    return schema

############################################################################ START - DATA CLEANING ############################################################################
def set_size(df): 
    df_size = df.count()
    return df_size

def data_cleaning(df):
    df = df.na.drop()
    df = df.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy/MM/dd HH:mm"))
    return df

def null_values(df):
    df = df.dropna()
    new_size = set_size(df)
    return new_size, df

def detecting_duplicates(df):
    grouped_transactions = df.groupBy(df.columns).count()
    number_of_duplicates = grouped_transactions.filter(col("count") > 1)\
        .agg((_sum('count')-count('count')).alias('number_of_duplicates')).toPandas().number_of_duplicates[0]
    
    if isnan(number_of_duplicates):
        number_of_duplicates = 0
        
    grouped_transactions = grouped_transactions.drop('count')
    return number_of_duplicates, grouped_transactions

def clean(df):
    original_size = set_size(df)
    print('Datafram size: {}\n'.format(original_size))

    df = data_cleaning(df)

    print('Finding null values... and transactions removal')
    new_size, df = null_values(df)
    print('Number of removed transactions with null values: {}\n'.format(original_size-new_size))

    print('Finding duplicated transactions... and transaction removal')
    number_of_duplicates, df = detecting_duplicates(df)
    print('Number of removed of duplicated transactions: {}\n'.format(number_of_duplicates))

    print('New dataframe size: {}'.format(new_size-number_of_duplicates))
    new_size=new_size-number_of_duplicates

    return df
############################################################################ END - DATA CLEANING ############################################################################

############################################################################ START - DATA ANALYSIS ############################################################################
# DATA ANALYSIS: ####### START - TREND OF THE AMOUNTS
def trend_amounts(df, df_type):
    df = df.select('timestamp','amount_received','amount_paid')
   
    group_per_day = df.groupBy(year("timestamp").alias("year"), 
                               month("timestamp").alias("month"), 
                               weekofyear("timestamp").alias("week"),
                               dayofmonth("timestamp").alias("day"))\
                                .agg(_sum("amount_received").alias("amount_received"), 
                                     _sum("amount_paid").alias("amount_paid")).orderBy('year','month','day')
    
    group_per_week = group_per_day.groupBy('year', 'month', 'week')\
        .agg(_sum("amount_received").alias("amount_received"), 
             _sum("amount_paid").alias("amount_paid")).orderBy('year','week')

    group_per_month = group_per_week.groupBy('year','month').agg(
        _sum('amount_received').alias('amount_received'),
        _sum('amount_paid').alias('amount_paid')).orderBy('year','month')

    df_pd_day = group_per_day.toPandas()
    df_pd_week = group_per_week.toPandas()
    df_pd_month = group_per_month.toPandas()
    
    plt.figure(figsize=(20, 6))
    draw_bar_line_plot(df_pd_day, 'day', 'Daily Transactions',131)
    draw_bar_line_plot(df_pd_week, 'week', 'Weekly Transactions',132)
    draw_bar_line_plot(df_pd_month, 'month', 'Monthly Transactions',133)
    plt.suptitle('{} - Amounts Grouped By Day, Week & Month'.format(df_type), fontsize=20)
    plt.tight_layout()
    plt.show()
    
def draw_bar_line_plot(df, grouped, title, n): 
    plt.subplot(n)
    plt.plot(df[grouped], df['amount_received'], label='Received', color='blue', marker='o', alpha=0.5)
    plt.plot(df[grouped], df['amount_paid'], label='Paid', color='red', marker='o', alpha=0.5)
    plt.xlabel(grouped.capitalize())
    plt.ylabel('Amount')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(df[grouped])
# DATA ANALYSIS: ####### END - TREND OF THE AMOUNTS

# DATA ANALYSIS: ####### START - TREND OF TRANSACTIONS PER DAY
def trend_transactions_x_day(df):
    df = df.select('timestamp','is_laundering')
    group_per_day = df.groupBy(year("timestamp").alias("year"), 
                               month("timestamp").alias("month"), 
                               weekofyear("timestamp").alias("week"),
                               dayofmonth("timestamp").alias("day"),
                               'is_laundering')\
                                .count()\
                                    .orderBy('year','month','day')
   
    laundering_df = group_per_day.filter('is_laundering==1').toPandas()
    non_laundering_df = group_per_day.filter('is_laundering==0').toPandas()
    
    # Raggruppa i dati per giorno e conta i valori di is_laundering e non_laundering
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharex=True)

    axes[0].plot(laundering_df['day'], laundering_df['count'], label='Laundering', marker='o', color='red')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Transactions')
    axes[0].set_title('Number Of Laundering Transactions Per Day')
    axes[0].legend()
    axes[0].grid(True)

    # Secondo subplot per "Non Laundering" (valore 0)
    axes[1].plot(non_laundering_df['day'], non_laundering_df['count'], label='NoN Laundering', marker='o', color='blue')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Transactions')
    axes[1].set_title('Number Of Non Laundering Transactions Per Day')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Number Of Transactions Per Day', fontsize=16)
    plt.tight_layout()
    plt.show()
# DATA ANALYSIS: ####### END - TREND OF TRANSACTIONS PER DAY

def get_percentage(total, n, df_type):
    print(df_type,': ','{}%'.format(n*100/total))
############################################################################ END - DATA ANALYSIS ############################################################################