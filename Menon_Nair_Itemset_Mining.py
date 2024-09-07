"""
CSCI-720: Big Data Analytics 
Project: Question 1 using Itemset Mining (Apriori Algorithm)
@author: Srivenkatesh Nair
         Abhinav Menon
"""


#Imported Libraries
import pandas as pd
import numpy as np
import math
from itertools import combinations

def generating_datamap(data):
    """
    This function does data cleaning and generates the data corresponding to contributing factors
    and vehicle type as a 2D List
    param: data: pandas dataframe which consists of actual datapoints
    return: factor_data: 2D list of actual data
            vehicle_data: 2D List of vehicle data
            factor_set: list of unique contributing factors
            vehicle_type_set: list of unique vehicle types
    """
    factor_data = []
    vehicle_data = []
    factor_set = set()
    vehicle_type_set = set()

    for _,df in data.iterrows():
        #Gathering data corresponding to Contributing Factor and Vehicle Type
        val1 = df['CONTRIBUTING FACTOR VEHICLES']
        val2 = df['VEHICLE TYPE CODES']
        val1 = val1.replace("'Unspecified'", 'Unspecified')
        val2 = val2.replace("'Unspecified'", 'Unspecified')
        val1 = val1[1:-1].split(',')
        val2 = val2[1:-1].split(',')

        #Did data remodeling here to check for 'nan' values
        for val in range(len(val1)):
            temp = val1[val].strip()
            if val1[val].strip() == 'nan':
                val1[val] = np.nan
            elif val1[val].strip() != 'Unspecified':
                val1[val] = temp[1:-1]
        
        for val in range(len(val2)):
            temp = val2[val].strip()
            if val2[val].strip() == 'nan':
                val2[val] = np.nan
            elif val2[val].strip() != 'Unspecified':
                val2[val] = temp[1:-1]
        
        #Computing the count of np.nan and 'Unspecified' value within each list of data
        count1 = 0
        for val in val1:
            if isinstance(val, float) and math.isnan(val):
                count1 += 1
            elif val == 'Unspecified' or val == ' Unspecified':
                count1 += 1
            else:
                if val not in factor_set:
                    factor_set.add(val)

        count2 = 0
        for val in val2:
            if isinstance(val, float) and math.isnan(val):
                count2 += 1
            elif val == 'Unspecified' or val == ' Unspecified':
                count2 += 1
            else:
                if val not in vehicle_type_set:
                    vehicle_type_set.add(val)
        #Dropping datapoints having only np.nan and 'Unspecified' as values within the list
        if count1 != 5 and count2 != 5:
            temp1 =[]
            for val in val1:
                if (isinstance(val, float) and math.isnan(val)) or val == 'Unspecified' or val == ' Unspecified':
                    continue
                else:
                    temp1.append(val)
            temp2 = []
            for val in val2:
                if (isinstance(val, float) and math.isnan(val)) or val == 'Unspecified' or val == ' Unspecified':
                    continue
                else:
                    temp2.append(val)
            factor_data.append(temp1)
            vehicle_data.append(temp2)

    return factor_data, vehicle_data, list(factor_set), list(vehicle_type_set)

def generating_possible_itemsets_k(data, k):
    """
    This function generates a list of all unique combinations of possible itemsets
    param: data: 2D list of Data
           k: length of each itemset
    return: itemsets: a list of all possible itemsets
    """
    itemsets = set()
    for value in data:
        itemsets.update(combinations(value, k))
    return list(itemsets)

def itemset_mining_k(itemsets, data, item_type):
    """
    This function performs our implementation of itemset mining algorithm
    param: itemsets: a list of all possible itemsets
           data: 2D list of datapoints
           item_type: 'vehicle' or 'factor'
    return: data_map: a dictionary consisting of results of itemset mining
    """
    #Minimum support of item_type
    if item_type == 'vehicle':
        min_support = 5
    elif item_type == 'factor':
        min_support = 2

    data_map = dict()
    #Iterating through each possible itemset
    for factors in itemsets:
        count = 0
        factor_list = list(factors)
        #Checking each datapoint
        for value in data:
            temp = 0
            #Doing item to item matching to update the count value
            for factor in range(len(factor_list)):
                if len(value) > factor:
                    if factor_list[factor] == value[factor]:
                        temp += 1
                    else: 
                        break
                    
                    if temp == len(factor_list):
                        count += 1
                
                else:
                    break
        #Checking if the count is greater than or equal to the minimum support      
        if count >= min_support:
            data_map[factors] = count
    sorted_datamap = dict(sorted(data_map.items(), key=lambda x: x[1], reverse = True))
    return sorted_datamap


def main():
    #Reading the data into a pandas database
    data_2019 = pd.read_csv('summer_2019_data.csv')
    data_2020 = pd.read_csv('summer_2020_data.csv')

    #Implementing data cleaning and generating the data as a 2D list
    factor_data1, vehicle_data1, factor_set1, vehicle_type1 = generating_datamap(data_2019)
    factor_data2, vehicle_data2, factor_set2, vehicle_type2 = generating_datamap(data_2020)
    
    #Length of each possible itemset
    k = 3

    #Implementing Itemset Mining for data corresponding to Summer 2019
    vehicle_list1 = generating_possible_itemsets_k(vehicle_data1, k)
    vehicle_map1 = itemset_mining_k(vehicle_list1, vehicle_data1,'vehicle' )
    max_set1 = set()
    count1 = 0
    check = 1
    print(f"\nITEMSET MINING FOR VEHICLE TYPE DATA (SUMMER 2019)")
    for key , value in vehicle_map1.items():
        print(f'{key} : {value}')
        if check == 1:
            max_set1 = key
            count1 = value
            check = 0
  
    factor_list1 = generating_possible_itemsets_k(factor_data1, k)
    factor_map1 = itemset_mining_k(factor_list1, factor_data1, 'factor')
    max_set2 = set()
    count2 = 0
    check = 1
    print(f"\nITEMSET MINING FOR CONTRIBUTING FACTORS DATA (SUMMER 2019)")
    for key , value in factor_map1.items():
        print(f'{key} : {value}')
        if check == 1:
            max_set2 = key
            count2 = value
            check = 0

    print(f"\nRESULT OF ITEMSET MINING (SUMMER 2019):")
    print(f'Most Frequent Vehicle Type in a 3 Vehicle Crash - Itemset: {max_set1}, Frequency: {count1}')
    print(f'Most Frequent Contributing Factors in a 3 Vehicle Crash - Itemset: {max_set2}, Frequency: {count2}\n')


    #Implementing Itemset Mining for data corresponding to Summer 2020
    vehicle_list2 = generating_possible_itemsets_k(vehicle_data2, k)
    vehicle_map2 = itemset_mining_k(vehicle_list2, vehicle_data2, 'vehicle')
    max_set3 = set()
    count3 = 0
    check = 1
    print(f"\nITEMSET MINING FOR VEHICLE TYPE DATA (SUMMER 2020)")
    for key , value in vehicle_map2.items():
        print(f'{key} : {value}')
        if check == 1:
            max_set3 = key
            count3 = value
            check = 0
  
    factor_list2 = generating_possible_itemsets_k(factor_data2, k)
    factor_map2 = itemset_mining_k(factor_list2, factor_data2, 'factor')
    max_set4 = set()
    count4 = 0
    check = 1
    print(f"\nITEMSET MINING FOR CONTRIBUTING FACTORS DATA (SUMMER 2020)")
    for key , value in factor_map2.items():
        print(f'{key} : {value}')
        if check == 1:
            max_set4 = key
            count4 = value
            check = 0

    
    print(f"\nRESULT OF ITEMSET MINING (SUMMER 2020):")
    print(f'Most Frequent Vehicle Type in a 3 Vehicle Crash - Itemset: {max_set3}, Frequency: {count3}')
    print(f'Most Frequent Contributing Factors in a 3 Vehicle Crash - Itemset: {max_set4}, Frequency: {count4}\n')





if __name__ == '__main__':
    main()