from imports import *

class MyGraph:
    # create the graph using the vertices and edges found in the dataset taken into account (train or test)
    def __init__(self, df):
        self.df = df
        self.ids = self.df.select('id','from_account','to_account')
        self.vertices, self.edges, self.g = self.create_graph()
        self.compute_inOut_degrees()

    def create_graph(self, init=True):
        vertices = self.df.select("from_account")\
                            .withColumnRenamed('from_account', 'id')\
                            .union(self.ids.select("to_account"))\
                            .distinct()
        if init:
            edges = self.df.withColumnRenamed('from_account', 'src')\
                .withColumnRenamed('to_account', 'dst')
        else:
            edges = self.df.withColumnRenamed('from_account', 'src')\
                .withColumnRenamed('to_account', 'dst').filter('from_account!=to_account and receiving_currency==payment_currency and payment_format="ACH"')\
                .select('id','timestamp','src','dst','payment_currency','payment_format')

        g = GraphFrame(vertices, edges)
        return vertices, edges, g

    def compute_inOut_degrees(self):
        # for each account, it computes the number of ingoing and outgoing transactions 
        vertexInDegrees = self.g.inDegrees
        vertexOutDegrees = self.g.outDegrees
        vertices = vertexInDegrees.join(vertexOutDegrees, 'id', 'fullouter').fillna(0)
        
        vertices = vertices.withColumnRenamed('id', 'from_account')
        self.ids = self.ids.alias('df').join(vertices.alias('vertices'), 'from_account', 'left')\
                        .withColumnRenamed('inDegree','from_account_inDegree')\
                        .withColumnRenamed('outDegree','from_account_outDegree')

        vertices = vertices.withColumnRenamed('from_account', 'to_account')
        self.ids = self.ids.join(vertices.alias('vertices'), 'to_account', 'left')\
                    .withColumnRenamed('inDegree','to_account_inDegree')\
                    .withColumnRenamed('outDegree','to_account_outDegree')

    def get_forwards(self):
        # it consists in getting all transactions in which the receiver of the transaction 
        # sends the same amount of received money to another account
        # OUTPUT: id of inolved transactions where:
        # - before_forward: 1 if a transaction is that one before a secondly forwarding transaction
        # - forward: 1 if a transaction is that one that makes the forward

        motif = "(a)-[e]->(b); (b)-[e2]->(c)"
        forwards = self.g.find(motif).filter("e.amount_received == e2.amount_paid and e.timestamp <= e2.timestamp and a!=b and b!=c")
    
        before_forward = forwards.select(col('e.id').alias('id'))\
            .distinct()\
            .withColumn('before_forward',lit(1))
        # distinct: I can use it, or I can count how many times the id is involved
        forward = forwards.select(col('e2.id').alias('id'))\
            .distinct()\
            .withColumn('forward',lit(1))
        # distinct: I can use it, or I can count how many times the id is involved
    
        self.forwards = before_forward.join(forward, 'id','left')#.na.fill(value=0,subset=['before_forward','forward'])
     
    def same_or_similar(self):
        # it search if for each transaction there is:
        # - another transaction with the same attributes, except the amounts (exists_same)
        # - another transaction with similar attributes, except the timestamps and amounts (exists_similar)
        motif = "(a)-[t1]->(b); (a)-[t2]->(b)"

        same_where = 't1.timestamp == t2.timestamp and \
                        t1.to_bank == t2.to_bank and \
                        t1.payment_currency == t2.payment_currency and \
                        t1.receiving_currency == t2.receiving_currency and \
                        t1.payment_format == t2.payment_format and \
                        t1.amount_paid != t2.amount_paid and \
                        t1.id != t2.id'
        
        self.same = self.g.find(motif).filter(same_where).select('t1.id').withColumn('exists_same',lit(1)).distinct()

        similar_where = 't1.timestamp != t2.timestamp and \
                        t1.to_bank == t2.to_bank and \
                        t1.payment_currency == t2.payment_currency and \
                        t1.receiving_currency == t2.receiving_currency and \
                        t1.payment_format == t2.payment_format and \
                        t1.amount_paid != t2.amount_paid'

        
        self.similar = self.g.find(motif).filter(similar_where).select('t1.id').withColumn('exists_similar',lit(1)).distinct()
       
########## START - FAN PATTERN ##########
    def compute_fan_in(self):
        """
            as explained in undestand_pattern.ipynb it is useful to compute the following feature: 
             - for each to_account, the number of incoming nodes to the same bank and all in node must have the same: 
                * receiving_currency 
                * payment_currency
                * payment_format
                 * there must be at most 4 days between the first transaction and the last in the series
        """
        motif = "(a)-[t1]->(b); (c)-[t2]->(b)"
        
        fan_in_query = 'abs(datediff(t1.timestamp, t2.timestamp)) <= 4 and \
                    t1.to_bank == t2.to_bank and \
                    t1.payment_currency == t2.payment_currency and \
                    t1.receiving_currency == t2.receiving_currency and \
                    t1.payment_format == t2.payment_format'
                
        fan_in = self.g.find(motif).filter(fan_in_query).select('a', 'b', 't1')
        fan_in = fan_in.groupBy('a', 'b', 't1').count().select('t1.id',col('count').alias('fan_in_degree'))

        return fan_in

    def compute_fan_out(self):
        """
            as explained in undestand_pattern.ipynb it is useful to compute the following feature: 
             - for each from_account, the number of outgoing nodes to the same bank and all in node must have the same: 
                * payment_format
                 * there must be at most 4 days between the first transaction and the last in the series
            
            in order to handle the big amount of data, data are firstly filtered:
            - self transaction (from_account == to_account) doesn't exist in the same fan-out
            - two similar transactions (t1(from_account, to_account) == t2(from_account, to_account) ) don't exist in the same fan-out 
            - fan-outs have ACH payment_format
        """
        _, _, g = self.create_graph(False)

        motif = "(a)-[t1]->(b); (a)-[t2]->(c)"
        
        fan_out_query = 'abs(datediff(t1.timestamp, t2.timestamp)) <= 4 and \
                        a != b and a != c and b != c and\
                        t1.id != t2.id'
                
        fan_out = g.find(motif).filter(fan_out_query).select('a', 'b', 'c', 't1.id')
        fan_out = fan_out.groupBy('a','b','c','id').count()
        fan_out = fan_out.groupBy('id').agg(count('*').alias('fan_out_degree')).select('id', 'fan_out_degree').withColumn('fan_out_degree', col('fan_out_degree')+1)
        
        return fan_out
    
    def compute_fan(self):
        fan_in = self.compute_fan_in()
        fan_out = self.compute_fan_out()  
        
        self.fans = fan_in.join(fan_out, 'id', 'fullouter')
########## END - FAN PATTERN ##########

########## START - CYCLE PATTERN ##########
    def generate_combinations(self,accounts):
        accounts = set(accounts)
        conditions = []
        for acc in accounts: 
            for acc2 in accounts:
                if acc!=acc2:
                    if ((acc, acc2) not in conditions) and ((acc2, acc) not in conditions):
                        conditions.append((acc,acc2))

        return ' and '.join(conditions[k][0]+'!='+conditions[k][1] for k in range(len(conditions)))

    def build_rules_of_cycles(self, max_iter):
        alphabet = list(map(chr, range(97, 123)))
        start = 2

        rules = []
        
        for i in range(start-1, max_iter+1):
            full_rule = []
            single_query = []
            transactions = []
            receiving_accounts = []
            select = []
            accounts = []
            for j in range(0, i+1):
                receiving_account = alphabet[j+1] if j < i else alphabet[0]
                receiving_accounts.append(receiving_account)

                single_transaction = 't{}'.format(j+1)
                transactions.append(single_transaction)

                starting_account = alphabet[j]

                single_rule = "({})-[{}]->({})".format(starting_account, single_transaction, receiving_account)
                accounts.append(starting_account)
                accounts.append(receiving_account)

                full_rule.append(single_rule)    
                select.append(single_transaction)
            
            single_query.append(' and '.join(transactions[k] + '.timestamp <= ' + transactions[k+1] + '.timestamp' for k in range(len(transactions) - 1)))
            single_query.append(self.generate_combinations(accounts))
        
            rules.append(('; '.join(full_rule), ' and '.join(single_query), select, (i+1)))

        return rules

    def find_cycles(self, max_iter):
        # this method obtains 4 features: 
        # - min_cycle: != 0 if the transaction is the starting one of a cycle
        # - max_cycle: != 0 if the transaction is the starting one of a cycle (== min_cycle if there's only that kind of degree)
        # - involved: 1 if the transaction is involved in a cycle, 0 otherwise
        _, _, g = self.create_graph(False)
        
        created_df = False

        max_iter = 1 if (max_iter-2) < 1 else max_iter-1
        rules = self.build_rules_of_cycles(max_iter)
        
        for rule in rules: 
            motif, query, select, degree = rule
            degree_cycle = g.find(motif).filter(query)
            
            for sel in range(len(select)): 
                if sel==0:
                    new_col = 'start'
                    select_id = '{}.id'.format(select[sel])
                else:
                    new_col = 't{}_id'.format(sel+1)
                    select_id = '{}.id'.format(select[sel])

                select[sel] = select_id
                degree_cycle = degree_cycle.select(select).withColumnRenamed('id', new_col)
                select[sel] = new_col

            degree_cycle_start = degree_cycle.select('start').distinct().withColumnRenamed('start', 'id')
            degree_cycle_involved = degree_cycle.drop('start')
            degree_cycle_involved = degree_cycle_involved.select(array([col(column) for column in degree_cycle_involved.columns])\
                                               .alias('id')).selectExpr('explode(id) as id').distinct()

            
            """
            This process asks a lot of time, so I should consider only the min, max degrees for the transaction that starts the cycle, 
            and 1 if a transaction is involved in a cycle, 0 otherwise.
        #     startings = self.in_cycle(degree_cycle_start, degree) # != 0 if a transaction is the starting one of a cycle
        #     intermediaries = self.in_cycle(degree_cycle_involved, degree, True) # != 0 if a transaction is involved in a cycle (not the starting one)

        #     print("adding {} degree...".format(degree))
        #     if not created_df: 
        #         starting_cycles = startings
        #         intermediaries_cycles = intermediaries
        #         created_df = True
        #     else:
        #         starting_cycles = starting_cycles.union(startings)
        #         intermediaries_cycles = intermediaries_cycles.union(intermediaries)

        # starting_cycles = starting_cycles.groupBy('id').agg(
        #     min("min_cycle").alias("min_cycle"),
        #     max("max_cycle").alias("max_cycle")
        # )
        # intermediaries_cycles = intermediaries_cycles.groupBy('id').agg(
        #     min("min_involved_cycle").alias("min_involved_cycle"),
        #     max("max_involved_cycle").alias("max_involved_cycle")
        # )

        # cycles = starting_cycles.join(intermediaries_cycles, 'id','fullouter')
        # self.df = self.df.join(cycles, 'id', 'left').na.fill(value=0,subset=['min_cycle', 'max_cycle', 'min_involved_cycle', 'max_involved_cycle'])
            So I used this solution below
            """
            startings = self.in_cycle(degree_cycle_start, degree) # != 0 if a transaction is the starting one of a cycle
            print("adding cycles of degree {}...".format(degree))
            if not created_df: 
                starting_cycles = startings
                intermediaries_cycles = degree_cycle_involved
                created_df = True
            else:
                starting_cycles = starting_cycles.union(startings)
                intermediaries_cycles = intermediaries_cycles.union(degree_cycle_involved)

        starting_cycles = starting_cycles.groupBy('id').agg(
            min("min_cycle").alias("min_cycle"),
            max("max_cycle").alias("max_cycle")
        )
        
        intermediaries_cycles = intermediaries_cycles.distinct()
        
        self.cycles = starting_cycles.join(intermediaries_cycles, 'id','fullouter').withColumn('involved', lit(1))
            
    def in_cycle(self, cycle_subset, degree, involved=False):
        columns = self.df.columns
        # if involved: 
        #     cycle_subset = cycle_subset\
        #         .withColumn('min_involved_cycle', lit(degree))\
        #         .withColumn('max_involved_cycle', lit(degree))
        # else:
        cycle_subset = cycle_subset\
            .withColumn('min_cycle', lit(degree))\
            .withColumn('max_cycle', lit(degree))
                
        return cycle_subset        
    
    """
        run time find_cycles + show:  
        * 2 degree ~ 22.9 sec
        * 3 degree ~ 20.2 sec
        * 4 degree ~ 23.6 sec
        * 5 degree ~ 35.5 sec
        * 6 degree ~ 37.2 sec
        * 7 degree ~ 46.4 sec
        * 8 degree ~ 52.7 sec
        * 9 degree ~ 58.3 sec
        * 10 degree ~ 1.22 min
        * 11 degree ~ 1.30 min
        * 12 degree ~ 3.15 min
    """
########## END - CYCLE PATTERN ########## 
       
    def join_ids(self):
        self.ids = self.ids.drop('from_account','to_account')
        self.ids = self.ids.join(self.forwards, 'id','left')
        self.ids = self.ids.join(self.similar, 'id', 'left').join(self.same, 'id', 'left')
        self.ids = self.ids.join(self.fans, 'id','left').na.fill(value=1,subset=['fan_out_degree', 'fan_in_degree'])
        self.ids = self.ids.join(self.cycles, 'id', 'left')
        self.ids = self.ids.na.fill(value=0,subset=['before_forward','forward','exists_same','exists_similar','min_cycle', 'max_cycle', 'involved'])

    def join_dataframe(self):
        self.df = self.df.join(self.ids, 'id', 'left').na.fill(value=0,subset=['before_forward','forward','exists_same','exists_similar','min_cycle', 'max_cycle', 'involved']).na.fill(value=1,subset=['fan_out_degree', 'fan_in_degree'])