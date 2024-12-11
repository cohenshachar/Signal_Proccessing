import matplotlib.pyplot as plt
import numpy as np
import math


def hw_function(x):
    return x*pow(math.e,x)  # given function

def hw_function_integral_over_area(left_x, right_x):
    return pow(math.e,right_x) - pow(math.e,left_x)

def ese(pdf, d_lst, r_lst):
    d_lst_length = len(d_lst)
    ese_sum = 0
    for x in range(len(pdf)):
        for i in range(1, d_lst_length):
            if x <= d_lst[i]:
                ese_sum += pow((x - int(r_lst[i - 1])), 2) * pdf[x]
                break
    return ese_sum

def ese_2(pdf, d_lst, r_lst,delta):
    d_lst_length = len(d_lst)
    ese_sum = 0
    i = 1
    for x in range(len(pdf)):  # calculate "center of mass" of the greyscales in each part and set decision levels
        x_real = x * delta + d_lst[0]
        if (x_real <= d_lst[i]):
            ese_sum += pow((hw_function(x_real) - hw_function(r_lst[i - 1])), 2) * pdf[x]
        else:
            i+=1
            x-=1
    return ese_sum

def lloyd_max(pdf, d_lst, e):
    d_lst_length = len(d_lst)
    r_lst = []
    for i in range(1,d_lst_length): #   calculate "center of mass" of the greyscales in each part and set decision levels
        top_sum, bot_sum = 0, 0
        for x in range(int(np.ceil(d_lst[i-1])), int(np.ceil(d_lst[i]))):
            top_sum += x * pdf[x]
            bot_sum += pdf[x]
        if bot_sum != 0:
            r_lst.append(top_sum/bot_sum)
        else: # there are no values in this part so we will put the r in the middle
            r_lst.append(d_lst[i-1] + (d_lst[i] - d_lst[i-1]) / 2)
    before_err = ese(pdf, d_lst, r_lst) #   calculate error
    for r in range(1,len(r_lst)):   #   set new reps based on the decision levels we calculated
        d_lst[r] = (r_lst[r-1] + r_lst[r])/2

    after_err = ese(pdf, d_lst,r_lst)
    if abs(before_err - after_err) < e:
        return r_lst
    return lloyd_max(pdf,d_lst,e) # again

def lloyd_max_2(pdf, d_lst, delta,e):
    r_lst = []
    top_sum, bot_sum = 0, 0
    i = 1
    for x in range(0,len(pdf)): #   calculate "center of mass" of the greyscales in each part and set decision levels
        x_real = x*delta + d_lst[0]
        if(x_real <= d_lst[i]):
            top_sum += x_real * pdf[x]
            bot_sum += pdf[x]
        else:
            if bot_sum != 0:
                r_lst.append(top_sum/bot_sum)
            else: # there are no values in this part so we will put the r in the middle
                r_lst.append(d_lst[i-1] + (d_lst[i] - d_lst[i-1]) / 2)
            top_sum, bot_sum = 0, 0
            i+=1
            x-=1
    if bot_sum != 0:
        r_lst.append(top_sum / bot_sum)
    else:  # there are no values in this part so we will put the r in the middle
        r_lst.append(d_lst[i - 1] + (d_lst[i] - d_lst[i - 1]) / 2)
    before_err = ese_2(pdf, d_lst, r_lst,delta) #   calculate error
    for r in range(1,len(r_lst)):   #   set new reps based on the decision levels we calculated
        d_lst[r] = (r_lst[r-1] + r_lst[r])/2
    after_err = ese_2(pdf, d_lst, r_lst,delta)
    if before_err - after_err < e:
        return r_lst
    return lloyd_max_2(pdf,d_lst,delta,e) # again


def find_median(lst_histogram):
    total = sum(lst_histogram)
    cur_sum = 0
    for i in range(len(lst_histogram)):
        if cur_sum > (total//2):
            return (i-1)
        elif cur_sum == total//2:
            return i
        cur_sum += lst_histogram[i]
    return len(lst_histogram)-1


def find_averages(lst_pdf):
    avg = 0
    for i in range(len(lst_pdf)):
        avg += i*lst_pdf[i]
    return int(avg)

#_________________________________ploting functions_________________________________-
def plot_rows(x_axis, func_mat, name):
    func_mat = np.array(func_mat)
    func_mat = np.hstack((func_mat, func_mat[:, -1].reshape(-1, 1)))
    i = 0
    for row in func_mat:
        name_i = str(name) + "_" + str(i)
        plt.figure(figsize=(7, 2.5))  # Adjust the figure size here
        plt.step(x_axis, row, where='post')
        plt.title(name_i)
        plt.grid(True)
        filename = f'{name_i}.jpg'
        plt.savefig(filename)
        plt.close()
        i += 1

def plot_cmp_rows(x1_axis,func_mat,x2_axis, cmp_fnc, name):
    func_mat = np.array(func_mat)
    func_mat = np.hstack((func_mat, func_mat[:, -1].reshape(-1, 1)))
    i = 0
    for row in func_mat:
        name_i = str(name) + "_" + str(i)
        plt.figure(figsize=(7, 2.5))  # Adjust the figure size here
        plt.step(x1_axis, row,x2_axis, cmp_fnc, where='post')
        plt.title(name_i)
        plt.grid(True)
        filename = f'{name_i}.jpg'
        plt.savefig(filename)
        plt.close()
        i += 1

def plot(x_axis, y_axis, name):
    y_axis.append(y_axis[-1])
    plt.figure(figsize=(7, 2.5))  # Adjust the figure size here
    plt.step(x_axis, y_axis, where='post')
    plt.title(name)
    plt.grid(True)
    name = name + "_plot"
    filename = f'{name}.jpg'
    plt.savefig(filename)
    y_axis.pop()
    plt.close()

def cmp_plot(x_axis, y_axis,x_axis2, y_axis2, name):
    y_axis.append(y_axis[-1])
    y_axis2.append(y_axis2[-1])
    plt.figure(figsize=(7, 2.5))  # Adjust the figure size here
    plt.step(x_axis, y_axis,x_axis2, y_axis2, where='post')
    plt.title(name)
    plt.grid(True)
    name = name + "_plot"
    filename = f'{name}.jpg'
    plt.savefig(filename)
    y_axis.pop()
    y_axis2.pop()
    plt.close()

#   _________________math operators__________________________-
def product(A,B):
    t_B = transpose(B)
    return [[round_err(sum([a_row[i]*b_col[i] for i in range(len(a_row))])) for b_col in t_B] for a_row in A]

def kronecker_product(A,B):
    k_product = []
    for a_row in A:
        product_row = []
        for a_i in a_row:
            block = []
            for b_row in B:
                block_row = []
                for b_i in b_row:
                    block_row.append(round_err(a_i*b_i))
                block.append(block_row)
            product_row.append(block)
        for i in range(len(product_row[0])):
            org_product_row = []
            for j in range(len(product_row)):
                for k in range(len(product_row[j][i])):
                    org_product_row.append(product_row[j][i][k])
            k_product.append(org_product_row)
    return k_product

def transpose(mat):
    res = [[] for _ in mat[0]]
    for row in mat:
        for i in range(len(row)):
            res[i].append(row[i])
    return res

def inner_product(A,B):
    t_A = transpose(A)
    return product(t_A,B)


#____________________________end of math operators _______________________________
#
#____________________________some helpful to the eye functions_________________________________
def print_matrix(mat):
    for row in mat:
        print(row)

def round_err(a):
    if  abs(round(a) - a) < 0.00000000000001:
        return round(a)
    else:
        return a