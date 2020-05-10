# -*- coding: utf-8 -*-
"""
Created: Thu Sep 26 2019
Author: Tamanna Baig
Program: Polynomial Regression

"""
#***********************Loading Libraries***************************#
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import csv
from statistics import mean, median

def plot_initial_datafile(file):
    plt.scatter(file.study, file.beer, c = file.gpa, marker = '*')
    plt.xlabel("Studying Minutes per Week")
    plt.ylabel("Beers Ounces per Week")
    plt.title("GPA of Student from Initial Data")
    plt.show()

#Randomize the data:
def data_randomize(data):
    return data.sample(frac=1, random_state = 6820).reset_index(drop=True)

#Normalize all the columns in the data:
def data_normalize(data, col_list = None):
    if col_list == None:
        data=(data-data.mean())/data.std()
    else:
        for col in col_list:
            data[col] = (data[col]-data[col].mean())/data[col].std()
    
    return(data)

#Split data into training set and test set:
def split_train_test(data):
    test_ratio = int(len(data) * 0.3)
    
    test_set = data.iloc[:test_ratio]
    train_set = data.iloc[test_ratio:]

    tr_add = open('baig_train_set.txt', 'a')
    tr_add.writelines(str(len(train_set))+'\n')
    tr_add.close()
    
    train_set.to_csv('baig_train_set.txt', sep = '\t' , index = False, header = False, mode = 'a')
    
    tes_add = open('baig_test_set.txt', 'a')
    tes_add.writelines(str(len(test_set))+'\n')
    tes_add.close()
    
    test_set.to_csv('baig_test_set.txt', sep = '\t' , index = False, header = False, mode = 'a')
    
    return train_set, test_set

#Calculate J values    
def calculate_cost(data, weight):
    m = len(data)
    cost_sum = 0
    
    for i, row in data.iterrows():
        x1 = row[0]
        x2 = row[1]
        y = row[2]
        
        cost_sum = cost_sum + ((hypothesis(x1, x2, weight) - y)**2)
        #c = ((1/(2*m)) * cost_sum)
        
    return ((1/(2*m)) * cost_sum)


def hypothesis(x1, x2, w_list):
    h = w_list[0] + (w_list[1] * x1) + (w_list[2] * x2) + (w_list[3] * x1 * x2) + (w_list[4] * (x1 ** 2)) + (w_list[5] * (x2 ** 2))
    
    return h

def get_z(wi, x1, x2):
    if wi == 0:
        z = 1
    elif wi == 1:
    	z = x1
    elif wi == 2:
    	z = x2
    elif wi == 3:
    	z = x1 * x2
    elif wi == 4:
    	z = x1**2
    elif wi == 5:
    	z = x2**2  
    
    return z

def update_weight(data, w_list, alpha = 0.01):
    upd_w = [None] * len(w_list)
    for wi in range(len(w_list)):
        m = len(data)
        temp = 0
        old_w = w_list[wi]
        for j, row in data.iterrows():
            x1 = row[0]
            x2 = row[1]
            y = row[2]
            
            z = get_z(wi, x1, x2)
            temp = temp + (hypothesis(x1, x2, w_list) - y ) * z
        
        new_w = old_w - alpha * (1/m) * temp
        
        upd_w[wi] = new_w
    
    return upd_w

def plot_final_J(iterations, cost_list):
    plt.scatter(iterations, cost_list)
    plt.xlabel("Iterations")
    plt.ylabel("Updated Costs J")
    plt.title("Cost J over Iterations")
    plt.savefig("baig_plot_final_J.png")
    plt.show()
    
def polynomial_regression(dataset, w_list, iterations = 50, alpha = 0.01):
    cost_list = list()
    iteration = list()
    
    for it in range(iterations):
        w_list = update_weight(dataset, w_list, alpha)
        cost = calculate_cost(dataset, w_list)
        cost_list.append(cost)
        iteration.append(it)
        
    df = pd.DataFrame({'Iteration_count': iteration, 'J_values': cost_list})

    df.to_csv('cost_over_iter.csv')
    #Plot of Final J Values
    #plot_final_J(iteration, cost_list)
        
    return w_list, cost_list[-1]

def make_predictions(data, weights):
    result = list()
    for i, row in data.iterrows():
        x1 = row[0]
        x2 = row[1]
        y = hypothesis(x1, x2, weights)
    
        result.append((y, row[2]))
    return result

def calculate_errors(predictions):
    err = list()
    for v in predictions:
        pred = v[0]
        actual = v[1]
        e = abs(pred - actual)/actual
        err.append(e)
    
    mean_err = mean(err)
    median_err = median(err)
    
    return mean_err, median_err 

def final_J_values(data, weights, final_J):
    pred_values = make_predictions(data, weights)
    mean_e, median_e = calculate_errors(pred_values)
    
    for i in range(len(weights)):
        print("w"+str(i)+": ", weights[i].round(6))
    print("-----------------")
    print("J: ", final_J)
    print("-----------------")
    print("Mean Error: ", mean_e.round(6))
    print("Median Error: ", median_e.round(6))

def get_col_data(data,col_names):
    means = {}
    stds = {}
    for col in col_names:
        means[col] = data[col].mean()
        stds[col] = data[col].std()
    return means, stds

def normalize_x(x, data_mean, data_std):
	xN = (x-data_mean)/data_std
	return xN

def get_user_inputs(cols_mean, cols_std, w_list):
    print("Enter Student Details:")
    user_x1 = int(input("Miuntes studied/Week: "))
    user_x2 = int(input("Ounces of Beer/Week: "))
    
    if user_x1 == 0 and user_x2 == 0:
        print("Terminating Program......")
        return
    
    user_x1 = normalize_x(user_x1, cols_mean['study_mins'], cols_std['study_mins'])
    user_x2 = normalize_x(user_x2, cols_mean['ounces_beer'], cols_std['ounces_beer'])
    
    print("\nFinal Weights: \n", w_list)
    y = hypothesis(user_x1, user_x2, w_list)
    
    print("\nThe predicted GPA of this student is: ", y.round(2))
    print("***************************************************************")
    get_user_inputs(cols_mean, cols_std, w_list)
    
def main():
    #*****************************Loading the data****************************#
    #Initial Values
    file_name = str(input("Enter the Input file name:  \n(For example - GPAData.txt)\n"))
    #file_name = 'GPAData.txt'
    alpha = 0.2
    w_list = [1] * 6
    it = 150
    
    col_names = ["study_mins", "ounces_beer", "gpa"]
    
    file = pd.read_csv(file_name, sep='\t', lineterminator='\n', skiprows = [0],
                header = None, names = col_names)
    
    #Call plot function to show GPA on basis of Study mins vs Ounces Beer
    #plot_initial_datafile(file)
    
    #Randomize the data
    rdm_data = data_randomize(file)
    #print(rdm_data.head())
    
    #Normalize the data
    norm_data = data_normalize(rdm_data, ['study_mins', 'ounces_beer'])
    #print(norm_data.head())
    
    # Split data into training and testing sets, save them as txts
    train_set, test_set = split_train_test(norm_data)
    
    
    w_list, final_J = polynomial_regression(train_set, w_list, it, alpha)
    
    #Calculating mean and std values for x1 and x2
    cols_mean, cols_std = get_col_data(file, ["study_mins", "ounces_beer"])
    
    #To print the Final Values
    #final_J_values(test_set, w_list, final_J)
    
    #To predict the GPA of a student by taking input from the user
    print("Program ready to accept input values!")
    print("Enter 0 for both values to exit!\n")
    get_user_inputs(cols_mean, cols_std, w_list)
    
#   #To calculate the J value on test set:
#    test_w_list, test_final_J = polynomial_regression(train_set, w_list, it, alpha)
#    print("Final J value on test set: ", test_final_J)
    
main()


