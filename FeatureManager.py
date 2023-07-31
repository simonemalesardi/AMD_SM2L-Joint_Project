from MyGraph import *
   
class FeatureManager:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        #self.take_equally()
        #self.split_train_test()

    def compute_features_of_whole_df(self):
        ach = StringIndexer(inputCol='payment_format', outputCol='pay_format').fit(self.dataframe)
        self.dataframe = ach.transform(self.dataframe)
        
        self.dataframe = self.dataframe\
            .withColumn('same_bank', (col('from_bank')==col('to_bank')).cast('integer'))\
            .withColumn('same_account', (col('from_account')==col('to_account')).cast('integer'))\
            .withColumn('same_currency', (col('receiving_currency')==col('payment_currency')).cast('integer'))\
            .withColumn('same_amounts', (col('amount_received')==col('amount_paid')).cast('integer'))
        
    def handle_dataset(self, train=True):
        if train: 
            self.train_graph = MyGraph(self.train_df)
            self.temp_ds = self.train_graph.ds
        else: 
            # + combine dataset
            subtracted = self.dataframe.subtract(self.sampled_dataset)
            self.test_df = self.test_df.union(subtracted)            
            # - combine dataset]
            self.test_graph = MyGraph(self.test_df)
            self.temp_ds = self.test_graph.ds

        self.temp_ds = self.ach.transform(self.temp_ds)
        self.temp_ds = self.temp_ds.drop('receiving_currency','payment_currency','payment_format','from_bank','to_bank',
                                         'from_account','to_account','timestamp','amount_received','amount_paid')

        self.transform_dataset()

        if train: 
            self.train_df = self.temp_ds
        else:
            self.test_df = self.temp_ds

    def take_equally(self):
        # classes are unbalanced, so we need to take a similar number of laundering and non laundering transactions:
        # in order to do that the dataset is filtered taking randomically a limited number of non laundering transactions 
        # and then laundering transactions are added 
        sampled_non_laundering = self.dataframe.filter('is_laundering==0').orderBy(rand()).limit(5200)
        self.sampled_dataset = sampled_non_laundering.union(self.dataframe.filter('is_laundering==1'))

    # it split the dataset into train and test set
    def split_train_test(self):
        self.train_df, self.test_df = self.sampled_dataset.randomSplit([0.8, 0.2])

    def transform_dataset(self):
        inputCols = self.temp_ds.drop('is_laundering').columns

        assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
        self.temp_ds = assembler.transform(self.temp_ds)
          
