from imports import *
from pyspark.sql.functions import col
from imports import *
from pyspark.sql.functions import col

class FeatureManager:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        #self.take_equally()
        #self.split_train_test()

    def compute_features_of_whole_df(self):
        # ach = StringIndexer(inputCol='payment_format', outputCol='pay_format').fit(self.dataframe)
        # self.dataframe = ach.transform(self.dataframe)
        
        cols = ('rec_cur','pay_cur','pay_for','f_b','t_b','f_a','t_a')

        self.dataframe = self.dataframe\
            .withColumnRenamed('receiving_currency', 'rec_cur')\
            .withColumnRenamed('payment_currency', 'pay_cur')\
            .withColumnRenamed('payment_format', 'pay_for')\
            .withColumnRenamed('from_bank', 'f_b')\
            .withColumnRenamed('to_bank', 't_b')\
            .withColumnRenamed('from_account', 'f_a')\
            .withColumnRenamed('to_account', 't_a')\

        currencies = self.dataframe.select('rec_cur').union(self.dataframe.select('pay_cur'))
        banks = self.dataframe.select('f_b').union(self.dataframe.select('t_b'))
        accounts = self.dataframe.select('f_a').union(self.dataframe.select('t_a'))

        currency = StringIndexer(inputCol='rec_cur', outputCol='receiving_currency')
        ach = StringIndexer(inputCol='pay_for', outputCol='payment_format')
        bank = StringIndexer(inputCol='f_b', outputCol='from_bank')
        account = StringIndexer(inputCol='f_a', outputCol='from_account')

        rec_currency_model = currency.fit(currencies)
        self.dataframe = rec_currency_model.transform(self.dataframe)
        pay_currency_model = rec_currency_model.setInputCol('pay_cur').setOutputCol('payment_currency')
        self.dataframe = pay_currency_model.transform(self.dataframe)
        
        from_bank_model = bank.fit(banks)
        self.dataframe = from_bank_model.transform(self.dataframe)
        to_bank_model = from_bank_model.setInputCol('t_b').setOutputCol('to_bank')
        self.dataframe = to_bank_model.transform(self.dataframe)
        
        from_account_model = account.fit(accounts)
        self.dataframe = from_account_model.transform(self.dataframe)
        to_account_model = from_account_model.setInputCol('t_a').setOutputCol('to_account')
        self.dataframe = to_account_model.transform(self.dataframe)

        payment_format_model = ach.fit(self.dataframe)
        self.dataframe = payment_format_model.transform(self.dataframe)
        self.ach_mapping = {v: k for k, v in dict(enumerate(payment_format_model.labels)).items()}

        column_order = ['id', 'timestamp',
                        'from_account','to_account','same_account',
                        'from_bank','to_bank','same_bank',
                        'amount_received','amount_paid','same_amounts',
                        'receiving_currency','payment_currency','same_currency',
                        'payment_format', 'is_laundering']
        
        self.dataframe = self.dataframe\
            .withColumn('same_bank', (col('from_bank')==col('to_bank')).cast('integer'))\
            .withColumn('same_account', (col('from_account')==col('to_account')).cast('integer'))\
            .withColumn('same_currency', (col('receiving_currency')==col('payment_currency')).cast('integer'))\
            .withColumn('same_amounts', (col('amount_received')==col('amount_paid')).cast('integer'))\
            .select(column_order).drop(*cols)

    
        
    # def handle_dataset(self, train=True):
    #     if train: 
    #         self.train_graph = MyGraph(self.train_df)
    #         self.temp_ds = self.train_graph.ds
    #     else: 
    #         # + combine dataset
    #         subtracted = self.dataframe.subtract(self.sampled_dataset)
    #         self.test_df = self.test_df.union(subtracted)            
    #         # - combine dataset]
    #         self.test_graph = MyGraph(self.test_df)
    #         self.temp_ds = self.test_graph.ds

    #     self.temp_ds = self.ach.transform(self.temp_ds)
    #     self.temp_ds = self.temp_ds.drop('receiving_currency','payment_currency','payment_format','from_bank','to_bank',
    #                                      'from_account','to_account','timestamp','amount_received','amount_paid')

    #     self.transform_dataset()

    #     if train: 
    #         self.train_df = self.temp_ds
    #     else:
    #         self.test_df = self.temp_ds

    # def take_equally(self):
    #     # classes are unbalanced, so we need to take a similar number of laundering and non laundering transactions:
    #     # in order to do that the dataset is filtered taking randomically a limited number of non laundering transactions 
    #     # and then laundering transactions are added 
    #     sampled_non_laundering = self.dataframe.filter('is_laundering==0').orderBy(rand()).limit(5200)
    #     self.sampled_dataset = sampled_non_laundering.union(self.dataframe.filter('is_laundering==1'))

    # # it split the dataset into train and test set
    # def split_train_test(self):
    #     self.train_df, self.test_df = self.sampled_dataset.randomSplit([0.8, 0.2])

    # def transform_dataset(self):
    #     inputCols = self.temp_ds.drop('is_laundering').columns

    #     assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    #     self.temp_ds = assembler.transform(self.temp_ds)
          
