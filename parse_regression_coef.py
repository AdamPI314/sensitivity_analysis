import pandas as pd
import numpy as np
import re

class parse_regression_coef_c:
    #read pathway data as pandas frame object
    @staticmethod
    def read_pathway_as_pandas_frame_object(filename):
        return pd.read_csv(filename, delimiter="\t", names=["number"])
    
    #convert pandas frame object to list
    @staticmethod
    def convert_pandas_frame_object_to_list(pathway_data_in):    
        # get rid of "#"
        mask = pathway_data_in.number.str.contains("#")
        pathway_data = pathway_data_in[~mask]        
        return np.array(pathway_data.number).reshape((-1, 4))
    
    #parse variance, zeroth_order, first_order and second_order values
    @staticmethod
    def var_zero_first_second(coef_list, ith, in_Nth_order_1st=5, in_Nth_order_2nd=2):
        coef_ith = coef_list[ith]
        var = float(coef_ith[0])
        zeroth = float(coef_ith[1])
        first_t = map(float, re.split(",", coef_ith[2]))
        first = np.array(first_t).reshape((-1, in_Nth_order_1st))
        second_t = map(float, re.split(",",coef_ith[3]))
        second = np.array(second_t).reshape((-1, in_Nth_order_2nd * in_Nth_order_2nd))
        return var, zeroth, first, second
    
    #parse variance, zeroth_order, first_order and second_order values
    #return zeroth_ordre, first_order and second_order in a list
    @staticmethod
    def var_zero_first_second_in_a_list(coef_list, ith):
        coef_ith = coef_list[ith]
        var = float(coef_ith[0])
        zeroth = float(coef_ith[1])
        first_t = [float(x) for x in re.split(",", coef_ith[2])]
        second_t = [float(x) for x in re.split(",",coef_ith[3])]
        return var, [zeroth] + first_t + second_t