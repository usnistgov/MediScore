
import re
#import detMetrics as dm
from itertools import product
from collections import OrderedDict
import pandas as pd
import numpy as np

class Partition:
    """This class represents a set of partitions for a single panda's dataframe,
       using one or several queries.
       It generates and stores each dataframe and their computed scores.
    """
    def __init__(self,dataframe,query,factor_mode,metrics,verbose=False): #,fpr_stop=1, isCI=False):
        """Constructor
        Attributes:
        - factor_mode : 'q' = single query
                        'qp' = cartesian product of the factors
                        '' = no factors. Average the entire frame
                        'qm' = no factors. Average the entire frame.
                               (Note that the frame should already be queried beforehand.)
        - factors_names : list of the dataframe's columns names
        - index_factor : Dictionary {'factor_name': index_of_the_column}
        - n_partitions : number of partitions generated
        - factor_dict : Dictionary associating each query's conditions to its factor
        - factors_order : List used to keep track of the factor's order in the query
        - part_query_list : List containing the single query in 'f' mode or
                                 containing each query generated query in 'fp' mode
                                 base on the cartesian product
        - part_values_list : List of the values' conditions for each factor for
                             each partitions
        - part_df_list : List of each partition's dataframe generated
        - part_metric_list: List of each partition's list of metrics generated

        - part_dm_list : List of each partition's DetMetric object generated
        """

        self.factor_mode = factor_mode
        self.factors_names = dataframe.columns.values
        self.index_factor = self.gen_index_factor(self.factors_names)
        self.task = dataframe['TaskID'].iloc[0]
        self.verbose = verbose

        if verbose: print("Beginning partition. Factor mode {} for query {}".format (factor_mode,query))
        # If we have a list of queries
        if (self.factor_mode == 'q'):
            #self.query = None
            self.part_query_list = query
            self.n_partitions = len(self.part_query_list)
        elif self.factor_mode == 'qp':
            self.query = query.replace(' ','')
            #TODO: Simplify the factors dictionary after the removing of the text render table
            self.factors_dict,self.factors_order = self.gen_factors_dict()
            self.part_values_list = self.gen_part_values_list()
            self.part_query_list = self.gen_part_query_list()
            self.n_partitions = len(self.part_values_list)
        elif (self.factor_mode == 'qm') or (self.factor_mode == ''):
            #self.query = None
            self.part_query_list = [query]
#            if self.factor_mode == 'qm':
#                self.part_query_list = [query]
#            else:
#                self.part_query_list = ['']
            self.n_partitions = len(self.part_query_list)

        self.part_df_list = self.gen_part_df_list(dataframe)
        self.part_metric_list = self.gen_part_metric_list(metrics)
#        self.part_dm_list = self.gen_part_dm_list(fpr_stop, isCI)


    def gen_index_factor(self,list_factors):
        """ Function used only in the constructor,
            shouldn't be called outside of the class.

            Generate the dictionary which make the association between
            each factor name and its column's index in the dataframe.
        """
        index_factor = dict()
        for i,factor in enumerate(list_factors):
            index_factor[factor] = i
        return index_factor

    def gen_factors_dict(self):
        """ Function used only in the constructor,
            shouldn't be called outside of the class.

            Parse the query and store it as a dictionary.
            {'factor_name': condition}
            The numerical factors are separeted from the general ones
            in a second dictionnary associated to the key 'Numerical_factors'
            For each numerical factors, the entire string (name+condition)
            is appended to a list associated to the key 'Numericals_factors_conditions'
        """
        L_factors = re.split('&|and',self.query)
        L_order = list()
        D_factors = OrderedDict()
        D_factors['Numericals_factors_conditions'] = list()
        D_factors['Numericals_factors'] = OrderedDict()
        for factor in L_factors:
            if '==' in factor:
                factor_name, factor_list = factor.split('==')
                D_factors[factor_name] = factor_list
                L_order.append(factor_name)
            elif '<' in factor:
                D_factors['Numericals_factors_conditions'].append(factor)
                split = re.split('[<|=]+',factor)
                if split[0] in self.factors_names:
                    D_factors['Numericals_factors'][split[0]] = factor
                    L_order.append(split[0])
                else:
                    D_factors['Numericals_factors'][split[1]] = factor
                    L_order.append(split[1])
            elif '>' in factor:
                D_factors['Numericals_factors_conditions'].append(factor)
                split = re.split('[>|=]+',factor)
                if split[0] in self.factors_names:
                    D_factors['Numericals_factors'][split[0]] = factor
                    L_order.append(split[0])
                else:
                    D_factors['Numericals_factors'][split[1]] = factor
                    L_order.append(split[1])

        return D_factors,L_order

    def gen_part_values_list(self):
        """ Function used only in the constructor,
            shouldn't be called outside of the class.

            This function computes the cartesian product of all the
            factors conditions, based on the factors dictionary
            It returns a list of lists. Each list contains the list
            of factors' conditions for one partition.
        """
        d = OrderedDict(self.factors_dict)
        L_values = list()
        L_num = d['Numericals_factors_conditions']
        del d['Numericals_factors_conditions']
        del d['Numericals_factors']
        for factor, values in d.items():
            L_values.append([factor+"=='"+x+"'" for x in eval(values)])
        L_product = list(map(list,product(*L_values)))
        [x.extend(L_num) for x in L_product]
        return L_product

    def gen_part_query_list(self):
        """ Function used only in the constructor,
            shouldn't be called outside of the class.

            This function generates the query associated the each
            factor's conditions list in the part_value_list.
        """
        List_part_query = list()
        for part_list in self.part_values_list:
            List_part_query.append(''.join([x+' & ' for x in part_list])[:-3])
        return List_part_query

    def gen_part_df_list(self,df):
        """ Function used only in the constructor,
            shouldn't be called outside of the class.

            This function computes and store each partition's dataframe
            generated according to its query in part_query_list.
        """
        df_list = list()
        if self.verbose: print("Generating dataframe for query(s) {}.".format(self.part_query_list))
        if (self.part_query_list == ['']) or (self.factor_mode == 'qm'):
            #base case
            return [df]

        for query in self.part_query_list:
#            df_list.append(df.query(query))

#                #TBD:while testing by each task (remove, add, clone), may need to drop_duplicates by the chosen column in the tf mode
#                operators = ['!=', '==']
#                if any(i in query for i in operators):
#                    chosenField = [x.strip() for x in query.replace('!=', '==').split('==')]
#                    new_df = sub_df.drop_duplicates('ProbeFileID', chosenField[0])

            try:
                sub_df = df.query(query)
            except pd.computation.ops.UndefinedVariableError:
                print("The query '{}' doesn't seem to refer to a valid key. Partitioning has failed. Please correct the query and try again.".format(query))
                exit(1)
            #print("Removing duplicates ...\n")
            new_df = sub_df.drop_duplicates('ProbeFileID') #Removing duplicates in case the data were merged by the JTmask metadata
            df_list.append(new_df)

        return df_list

    #def gen_part_dm_list(self, fpr_stop, isCI):
    def gen_part_metric_list(self,metrics):
        """ Function used only in the constructor,
            shouldn't be called outside of the class.

            This function creates and store each partition's metric
            dataframe according to its dataframe in part_df_list.
        """
        if (self.factor_mode == '') or (self.factor_mode == 'qm'):
            return self.part_df_list

        dm_list = list()

        for df, query in zip(self.part_df_list,self.part_query_list):
            if not df.empty:
                if self.verbose: print("Current query: {}".format(query))
                #dm_list.append(df[metrics])
                #dm_list.append(dm.detMetrics(df['ConfidenceScore'], df['IsTarget'],fpr_stop, isCI))
            else:
                print('#### Warning: Empty DataFrame for query "{}"\n#### Please verify factors conditions.'.format(query))
            if self.verbose: print("Metrics: {}".format(df[metrics]))
            dm_list.append(df[metrics])
        return dm_list

    def get_query(self):
        """ Getter for the query attributes, return a formatted version
            for the query string.
        """
        return self.query.replace('&',' & ')\
                         .replace('and',' and ')\
                         .replace('==',' == ')\
                         .replace('<',' < ')\
                         .replace('>',' > ')

    #def render_table(self):
    def render_table(self,metrics):
        """ This function computes a table (as a dataframe) for each partition
            containing the informations stored in the corresponding object.
            It returns a list of dataframes in 'f' mode and
            one dataframe listing all the partitions in 'fp' mode
        """

        if self.verbose: print("Rendering table for metrics {}...".format(metrics))
        def find_factor_list_pos(List, factor):
            i = 0
            while factor not in List[i]: i += 1
            return i

        if (self.factor_mode == '') or (self.factor_mode == 'qm'):
            #base case
            data = dict()
            dm = self.part_metric_list[0]
            metsplus = []
            for m in metrics:
                data[m] = [dm[m].mean(skipna=True)]
                metsplus.append(m)
                #add standard deviations too if length of data not nan is > 1
                if dm[m].count() > 1:
                    stdname = '_'.join(['stddev',m])
                    data[stdname] = [dm[m].std(skipna=True)]
                    metsplus.append(stdname)
            data['TaskID'] = [self.task]
            data['Query'] = self.part_query_list
            if data['Query'] == ['']:
                data['Query'] = ['Full']
            columns = ['TaskID','Query']
            columns.extend(metsplus)
            return [pd.DataFrame(data=data,columns=columns)]

        if self.factor_mode == 'q':
            df_list = list()
            for i,query in enumerate(self.part_query_list):
                #dm = self.part_dm_list[i]
                dm = self.part_metric_list[i]
                if len(dm) == 0:
                    #skip where dm is empty
                    continue
                data = {'Query': query}
                metsplus = []
                for m in metrics:
                    data[m] = dm[m].mean(skipna=True) #average this
                    metsplus.append(m)
                    #add standard deviations too if length of data not nan is > 1
                    if dm[m].count() > 1:
                        stdname = '_'.join(['stddev',m])
                        data[stdname] = [dm[m].std(skipna=True)]
                        metsplus.append(stdname)

#                data = {'Query': query,
#                         'auc': dm.auc,
#                         'fpr_stop' : dm.fpr_stop,
#                         'eer':dm.eer,
#                         'auc_ci_lower':dm.ci_lower,
#                         'auc_ci_upper':dm.ci_upper}
#                columns = ['Query','auc','fpr_stop','eer','auc_ci_lower','auc_ci_upper']
                data['TaskID'] = [self.task]
                columns = ['TaskID','Query']
                columns.extend(metsplus)
                df_list.append(pd.DataFrame(data=data,columns=columns))
            return df_list

        elif self.factor_mode == 'qp':
            data = dict()
            # Looking for the values of each fields
            for m in metrics:
                data[m] = []

#            data = {'auc': [],'fpr_stop': [],'eer': [],'auc_ci_lower': [], 'auc_ci_upper': []}
            n_emptydf = 0
            metsplus = []
            for i,partition in enumerate(self.part_values_list):
                dm = self.part_metric_list[i]
                if len(dm) == 0:
                    #skip where dm is empty
                    n_emptydf = n_emptydf + 1
                    continue
                for field in self.factors_order:
                    full_condition = partition[find_factor_list_pos(partition,field)]
                    if '==' in full_condition:
                        _,condition = full_condition.split('==')
                    else: condition = full_condition

                    if field not in data:
                        data[field] = [condition]
                    else:
                        data[field].append(condition)

#                dm = self.part_dm_list[i]
                for m in metrics:
                    data[m] = dm[m].mean(skipna=True) #average this
                    metsplus.append(m)
                    #add standard deviations too if length of data not nan is > 1
                    if dm[m].count() > 1:
                        stdname = '_'.join(['stddev',m])
                        data[stdname] = [dm[m].std(skipna=True)]
                        if stdname not in metsplus:
                            metsplus.append(stdname)

#                data['auc'].append(dm.auc)
#                data['fpr_stop'].append(dm.fpr_stop)
#                data['eer'].append(dm.eer)
#                data['auc_ci_lower'].append(dm.ci_lower)
#                data['auc_ci_upper'].append(dm.ci_upper)
            data['TaskID'] = self.task
            columns = ['TaskID']
            columns.extend(self.factors_order)
            columns.extend(metsplus)
#            columns.extend(['auc','fpr_stop','eer','auc_ci_lower', 'auc_ci_upper'])
            index = ['Partition_'+str(i) for i in range(self.n_partitions-n_emptydf)]

            df = pd.DataFrame(data,index,columns)
            return [df]

    def __repr__(self):
        """Representation method
        """
        return "Partition :\nQuery = {}\nList partitions values =\n{}\nList partitions queries =\n{}\n"\
                .format(self.get_query(),self.part_values_list,self.part_query_list)



if __name__ == '__main__':

    def gen_dataframe(n_rows):
        """Test function that generates a random dataframe of n_rows rows,
           containing various datatypes (strings, integers, floats, boolean)
        """
        import numpy as np
        import pandas as pd
        set_f1 = 'Uppercase_4'
        set_f2 = 'Lowercase_7'
        set_f3 = 'Integers_10'
        set_f4 = 'Boolean'
        set_f5 = 'RandomFloat'
        list_factors_values = [set_f1, set_f2, set_f3, set_f4, set_f5]
        data_dict = dict()
        for i,factor_values in enumerate(list_factors_values):
            if '_' in factor_values:
                Type_value, n = factor_values.split('_')
                if Type_value == 'Uppercase':
                    data_dict['factor_' + str(i)] = np.random.randint(65,65+int(n), size=n_rows).astype(np.uint32).view('U1')
                elif Type_value == 'Lowercase':
                    data_dict['factor_' + str(i)] = np.random.randint(97,97+int(n), size=n_rows).astype(np.uint32).view('U1')
                elif Type_value == 'Integers':
                    data_dict['factor_' + str(i)] = np.random.randint(int(n), size=n_rows)
            elif 'Boolean' == factor_values:
                nb = -(-n_rows // 8)     # ceiling division
                b = np.fromstring(np.random.bytes(nb), np.uint8, nb)
                data_dict['factor_' + str(i)] = np.unpackbits(b)[:n_rows].view(np.bool)
            elif 'RandomFloat' == factor_values:
                data_dict['factor_' + str(i)] = np.random.rand(n_rows)
        return pd.DataFrame(data_dict)

    #path = "./"
    query = "factor_0 == ['A','C'] & factor_1 == ['a','b'] & 2<=factor_2<=4 & factor_4<10000"
    df = gen_dataframe(100)
    mypart = Partition(df,query)
#    mypart.render_table(path)
