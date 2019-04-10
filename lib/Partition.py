
import re
import detMetrics as dm
from itertools import product
from collections import OrderedDict
import pandas as pd


class Partition:
    """This class represents a set of partitions for a single panda's dataframe,
       using one or several queries.
       It generates and stores each dataframe and their corresponding
       DetMetric objects.
    """

    def __init__(self, dataframe, query, factor_mode, fpr_stop=1, isCI=False, ciLevel=0.9, dLevel=0.0, total_num=1, sys_res='all', overlap_cols=['ProbeFileID']):
        """Constructor
        Attributes:
        - factor_mode : 'q' = single query
                        'qp' = cartesian product of the factors for partitioning data
                        'qm' = filtering for target trials for selective manipulation
        - factors_names : list of the dataframe's columns names
        - index_factor : Dictionnary {'factor_name': index_of_the_column}
        - n_partitions : number of partitions generated
        - factor_dict : Dictionnary associating each query's conditions to its factor
        - factors_order : List used to keep track of the factor's order in the query
        - part_query_list : List containing the single query in 'q' and 'qm' mode or
                                 containing each query generated query in 'qp' mode
                                 base on the cartesian product
        - part_values_list : List of the values' conditions for each factor for
                             each partitions
        - part_df_list : List of each partition's dataframe generated
        - part_dm_list : List of each partition's DetMetric object generated
        """

        self.factor_mode = factor_mode
        self.factors_names = dataframe.columns.values
        self.index_factor = self.gen_index_factor(self.factors_names)
        #self.task = task
        self.overlap_cols = overlap_cols

        # If we have a list of queries
        if self.factor_mode == 'q' or self.factor_mode == 'qm':
            #self.query = None
            self.part_query_list = query
            self.n_partitions = len(self.part_query_list)
        elif self.factor_mode == 'qp':
            self.query = query.replace(' ', '')
            # TODO: Simplify the factors dictionnary after the removing of the text render table
            self.factors_dict, self.factors_order = self.gen_factors_dict()
            self.part_values_list = self.gen_part_values_list()
            self.part_query_list = self.gen_part_query_list()
            self.n_partitions = len(self.part_values_list)

        self.part_df_list = self.gen_part_df_list(dataframe)
        self.part_dm_list = self.gen_part_dm_list(
            fpr_stop, isCI, ciLevel, dLevel, total_num, sys_res)

    def gen_index_factor(self, list_factors):
        """ Function used only in the constructor,
            should'nt be called outside of the class.

            Generate the dictionnary which make the association between
            each factor name and its column's index in the dataframe.
        """
        index_factor = dict()
        for i, factor in enumerate(list_factors):
            index_factor[factor] = i
        return index_factor

    def gen_factors_dict(self):
        """ Function used only in the constructor,
            should'nt be called outside of the class.

            Parse the query and store it as a dictionnary.
            {'factor_name': condition}
            The numerical factors are separeted from the general ones
            in a second dictionnary associated to the key 'Numerical_factors'
            For each numerical factors, the entire string (name+condition)
            is appended to a list associated to the key 'Numericals_factors_conditions'
        """
        L_factors = re.split('&|and', self.query)
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
                split = re.split('[<|=]+', factor)
                if split[0] in self.factors_names:
                    D_factors['Numericals_factors'][split[0]] = factor
                    L_order.append(split[0])
                else:
                    D_factors['Numericals_factors'][split[1]] = factor
                    L_order.append(split[1])
        return D_factors, L_order

    def gen_part_values_list(self):
        """ Function used only in the constructor,
            should'nt be called outside of the class.

            This function computes the cartesian product of all the
            factors conditions, based on the factors dictionnary
            It returns a list of lists. Each list contains the list
            of factors' conditions for one partition.
        """
        d = OrderedDict(self.factors_dict)
        L_values = list()
        L_num = d['Numericals_factors_conditions']
        del d['Numericals_factors_conditions']
        del d['Numericals_factors']
        for factor, values in d.items():
            L_values.append([factor + "=='" + x + "'" for x in eval(values)])
        L_product = list(map(list, product(*L_values)))
        [x.extend(L_num) for x in L_product]
        return L_product

    def gen_part_query_list(self):
        """ Function used only in the constructor,
            should'nt be called outside of the class.

            This function generates the query associated the each
            factor's conditions list in the part_value_list.
        """
        List_part_query = list()
        for part_list in self.part_values_list:
            List_part_query.append(''.join([x + ' & ' for x in part_list])[:-3])
        return List_part_query

    def gen_part_df_list(self, df):
        """ Function used only in the constructor,
            should'nt be called outside of the class.

            This function computes and store each partition's dataframe
            generated according to its query in part_query_list.
        """
        df = df.fillna("")
        df_list = list()
        for query in self.part_query_list:
            #            df_list.append(df.query(query))
            if self.factor_mode == 'qm':
                # testing as the manipulation task
                query = "(" + query + " and IsTarget == ['Y']) or IsTarget == ['N']"
                print("Query for target trials: {}".format(query))

#                #TBD:while testing by each task (remove, add, clone), may need to drop_duplicates by the chosen column in the tf mode
#                operators = ['!=', '==']
#                if any(i in query for i in operators):
#                    chosenField = [x.strip() for x in query.replace('!=', '==').split('==')]
#                    new_df = sub_df.drop_duplicates('ProbeFileID', chosenField[0])

            sub_df = df.query(query)
            new_df = sub_df.drop_duplicates(subset=self.overlap_cols)
            #print("sub_df data {}".format(sub_df))
            #print("sub_df data size {}".format(sub_df.shape))
            #print("Removing duplicates ...\n")
            # Removing duplicates in case the data were merged by the JTmask metadata,
            # not for splice
            # if self.task == 'manipulation':
            #     new_df = sub_df.drop_duplicates('ProbeFileID')
            # elif self.task == 'splice':
            #     new_df = sub_df.drop_duplicates(subset=['ProbeFileID', 'DonorFileID'])
            # elif self.task == 'eventverification':
            #     new_df = sub_df.drop_duplicates(subset=['ProbeFileID', 'EventName'])
            # elif self.task == 'camera':
            #     new_df = sub_df.drop_duplicates(subset=['ProbeFileID', 'HDDeviceID'])

            #print("new_df data size {}".format(new_df.shape))
            df_list.append(new_df)

        return df_list

    def gen_part_dm_list(self, fpr_stop, isCI, ciLevel, dLevel, total_num, sys_res):
        """ Function used only in the constructor,
            should'nt be called outside of the class.

            This function creates and store each partition's detMetric
            object according to its dataframe in part_df_list.
        """
        dm_list = list()
        for df, query in zip(self.part_df_list, self.part_query_list):
            if not df.empty:
                print("Current query: {}".format(query))
                dm_list.append(dm.detMetrics(
                    df['ConfidenceScore'], df['IsTarget'], fpr_stop, isCI, ciLevel, dLevel, total_num, sys_res))
            else:
                print(
                    '#### Error: Empty DataFrame for this query "{}"\n#### Please verify factors conditions.'.format(query))
        return dm_list

    def get_query(self):
        """ Getter for the query attributes, return a formatted version
            for the query string.
        """
        return self.query.replace('&', ' & ')\
                         .replace('and', ' and ')\
                         .replace('==', ' == ')\
                         .replace('<', ' < ')

    def render_table(self):
        """ This function compute a table (as a dataframe) for each partitions
            containing the informations stored in the corresponding DetMetric object.
            It returns a list of dataframe in 'q' and 'qm' mode and
            one dataframe listing all the partitions in 'qp' mode
        """

        def find_factor_list_pos(List, factor):
            i = 0
            while factor not in List[i]:
                i += 1
            return i

        if self.factor_mode == 'q' or self.factor_mode == 'qm':
            df_list = list()
            for i, query in enumerate(self.part_query_list):
                dm = self.part_dm_list[i]
                data = {'QUERY': query,
                        'TRR': dm.trr,
                        'SYS_RESPONSE': dm.sys_res,
                        'AUC': dm.auc,
                        'EER': dm.eer,
                        'FAR_STOP': dm.fpr_stop,
                        'AUC@FAR': dm.auc_at_fpr,
                        'CDR@FAR': dm.tpr_at_fpr,
                        'CI_LEVEL': dm.ci_level,
                        'AUC_CI_LOWER': dm.auc_ci_lower,
                        'AUC_CI_UPPER': dm.auc_ci_upper,
                        'AUC_CI_LOWER@FAR': dm.auc_at_fpr_ci_lower,
                        'AUC_CI_UPPER@FAR': dm.auc_at_fpr_ci_upper,
                        'CDR_CI_LOWER@FAR': dm.tpr_at_fpr_ci_lower,
                        'CDR_CI_UPPER@FAR': dm.tpr_at_fpr_ci_upper
                        }
                index = ['Q' + str(i)]
                columns = ['QUERY', 'TRR', 'SYS_RESPONSE', 'AUC', 'EER', 'FAR_STOP', 'AUC@FAR', 'CDR@FAR', 'CI_LEVEL','AUC_CI_LOWER','AUC_CI_UPPER', 'AUC_CI_LOWER@FAR','AUC_CI_UPPER@FAR', 'CDR_CI_LOWER@FAR', 'CDR_CI_UPPER@FAR']
                df_list.append(pd.DataFrame(data, index, columns).round(6))
            return df_list

        elif self.factor_mode == 'qp':
            data = dict()
            # Looking for the values of each fields
            data = {'TRR': [], 'SYS_RESPONSE': [], 'AUC': [], 'EER': [], 'FAR_STOP': [], 'AUC@FAR': [], 'CDR@FAR': [], 'AUC_CI_LOWER': [],'AUC_CI_UPPER': [], 'AUC_CI_LOWER@FAR': [],'AUC_CI_UPPER@FAR': [], 'CDR_CI_LOWER@FAR': [], 'CDR_CI_UPPER@FAR': []}
            for i, partition in enumerate(self.part_values_list):
                for field in self.factors_order:
                    full_condition = partition[find_factor_list_pos(partition, field)]
                    if '==' in full_condition:
                        _, condition = full_condition.split('==')
                    else:
                        condition = full_condition

                    if field not in data:
                        data[field] = [condition]
                    else:
                        data[field].append(condition)

                dm = self.part_dm_list[i]
                data['TRR'].append(dm.trr)
                data['SYS_RESPONSE'].append(dm.sys_res)
                data['AUC'].append(dm.auc)
                data['EER'].append(dm.eer)
                data['FAR_STOP'].append(dm.fpr_stop)
                data['AUC@FAR'].append(dm.auc_at_fpr)
                data['CDR@FAR'].append(dm.tpr_at_fpr)
                data['CI_LEVEL'].append(dm.ci_level)
                data['AUC_CI_LOWER'].append(dm.auc_ci_lower)
                data['AUC_CI_UPPER'].append(dm.auc_ci_upper)
                data['AUC_CI_LOWER@FAR'].append(dm.auc_at_fpr_ci_lower)
                data['AUC_CI_UPPER@FAR'].append(dm.auc_at_fpr_ci_upper)
                data['CDR_CI_LOWER@FAR'].append(dm.tpr_at_fpr_ci_lower)
                data['CDR_CI_UPPER@FAR'].append(dm.tpr_at_fpr_ci_upper)

            columns = list(self.factors_order)
            columns.extend(['TRR', 'SYS_RESPONSE', 'AUC', 'EER', 'FAR_STOP', 'AUC@FAR', 'CDR@FAR', 'CI_LEVEL','AUC_CI_LOWER','AUC_CI_UPPER', 'AUC_CI_LOWER@FAR','AUC_CI_UPPER@FAR', 'CDR_CI_LOWER@FAR', 'CDR_CI_UPPER@FAR'])
            index = ['P' + str(i) for i in range(self.n_partitions)]
            df = pd.DataFrame(data, index, columns).round(6)
            return df

    def __repr__(self):
        """Representation method
        """
        return "Partition :\nQuery = {}\nList partitions values =\n{}\nList partitions queries =\n{}\n"\
            .format(self.get_query(), self.part_values_list, self.part_query_list)


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
        for i, factor_values in enumerate(list_factors_values):
            if '_' in factor_values:
                Type_value, n = factor_values.split('_')
                if Type_value == 'Uppercase':
                    data_dict['factor_' + str(i)] = np.random.randint(65,
                                                                      65 + int(n), size=n_rows).astype(np.uint32).view('U1')
                elif Type_value == 'Lowercase':
                    data_dict['factor_' + str(i)] = np.random.randint(97,
                                                                      97 + int(n), size=n_rows).astype(np.uint32).view('U1')
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
    mypart = Partition(df, query)
#    mypart.render_table(path)
