
import re
import detMetrics as dm
from itertools import product
from collections import OrderedDict
import pandas as pd

class Partition:

    def __init__(self,dataframe,query,factor_mode,fpr_stop=1, isCI=False):
        self.factor_mode = factor_mode
        self.factors_names = dataframe.columns.values
        self.index_factor = self.gen_index_factor(self.factors_names)

        # If we have a list of queries
        if factor_mode == 'f':
            #self.query = None
            self.part_query_list = query
            self.n_partitions = len(self.part_query_list)
        elif factor_mode == 'fp':
            self.query = query.replace(' ','')
            #TODO: Simplify the factors dictionnary after the removing of the text render table
            self.factors_dict,self.factors_order = self.gen_factors_dict()
            self.part_values_list = self.gen_part_values_list()
            self.part_query_list = self.gen_part_query_list()
            self.n_partitions = len(self.part_values_list)

        self.part_df_list = self.gen_part_df_list(dataframe)
        self.part_dm_list = self.gen_part_dm_list(fpr_stop, isCI)


    def gen_index_factor(self,list_factors):
        index_factor = dict()
        for i,factor in enumerate(list_factors):
            index_factor[factor] = i
        return index_factor

    def gen_factors_dict(self):
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
        return D_factors,L_order

    def gen_part_values_list(self):
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
        List_part_query = list()
        for part_list in self.part_values_list:
            List_part_query.append(''.join([x+' & ' for x in part_list])[:-3])
        return List_part_query

    def gen_part_df_list(self,df):
        df_list = list()
        for query in self.part_query_list:
            df_list.append(df.query(query))
        return df_list

    def gen_part_dm_list(self, fpr_stop, isCI):
        dm_list = list()
        for df, query in zip(self.part_df_list,self.part_query_list):
            if not df.empty:
                print("Current query = {}".format(query))
                dm_list.append(dm.detMetrics(df['ConfidenceScore'], df['IsTarget'],fpr_stop, isCI))
            else:
                print('#### Error: Empty DataFrame for this query "{}"\n#### Please verify factors conditions.'.format(query))
        return dm_list

    def get_query(self):
        return self.query.replace('&',' & ')\
                         .replace('and',' and ')\
                         .replace('==',' == ')\
                         .replace('<',' < ')

    def render_table(self):

        def find_factor_list_pos(List, factor):
            i = 0
            while factor not in List[i]: i += 1
            return i

        if self.factor_mode == 'f':
            df_list = list()
            for i,query in enumerate(self.part_query_list):
                dm = self.part_dm_list[i]
                data = {'Query': query,
                         'auc': dm.auc,
                         'fpr_stop' : dm.fpr_stop,
                         'eer':dm.eer,
                         'auc_ci_lower':dm.ci_lower,
                         'auc_ci_upper':dm.ci_upper}
                index = ['P:']
                columns = ['Query','auc','fpr_stop','eer','auc_ci_lower','auc_ci_upper']
                df_list.append(pd.DataFrame(data,index,columns))
            return df_list

        elif self.factor_mode == 'fp':
            data = dict()
            # Looking for the values of each fields
            data = {'auc': [],'fpr_stop': [],'eer': [],'auc_ci_lower': [], 'auc_ci_upper': []}
            for i,partition in enumerate(self.part_values_list):
                for field in self.factors_order:
                    full_condition = partition[find_factor_list_pos(partition,field)]
                    if '==' in full_condition:
                        _,condition = full_condition.split('==')
                    else: condition = full_condition

                    if field not in data:
                        data[field] = [condition]
                    else:
                        data[field].append(condition)

                dm = self.part_dm_list[i]
                data['auc'].append(dm.auc)
                data['fpr_stop'].append(dm.fpr_stop)
                data['eer'].append(dm.eer)
                data['auc_ci_lower'].append(dm.ci_lower)
                data['auc_ci_upper'].append(dm.ci_upper)

            columns = list(self.factors_order)
            columns.extend(['auc','fpr_stop','eer','auc_ci_lower', 'auc_ci_upper'])
            index = ['Partition_'+str(i) for i in range(self.n_partitions)]
            df = pd.DataFrame(data,index,columns)
            return df

    def __repr__(self):
        return "Partition :\nQuery = {}\nList partitions values =\n{}\nList partitions queries =\n{}\n"\
                .format(self.get_query(),self.part_values_list,self.part_query_list)



if __name__ == '__main__':

    def gen_dataframe(n_rows):
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

    #path = "/Users/tnk12/Documents/ProjetD/MyProject/"
    query = "factor_0 == ['A','C'] & factor_1 == ['a','b'] & 2<=factor_2<=4 & factor_4<10000"
    df = gen_dataframe(100)
    mypart = Partition(df,query)
#    mypart.render_table(path)
