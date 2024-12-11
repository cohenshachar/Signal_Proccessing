from math import sqrt
import math
import numpy as np
import util as ut

sqrt_2 = sqrt(2)
H_2 = [[1,1],[1,-1]]
H_sqrt2 = ut.kronecker_product([[1/sqrt_2]],H_2)

def create_2_to_nth_I(n):
    dim = pow(2,n)
    I_matrix = [[int(i == j) for j in range(dim)] for i in range(dim)]
    return I_matrix

def create_nth_Hadamard(n):
    if n == 1:
        return H_sqrt2
    return ut.kronecker_product(H_sqrt2, create_nth_Hadamard(n-1))

def create_nth_Walsh_Hadamard(hadamard_matrix):
    walsh_matrix = [[] for _ in hadamard_matrix]
    for row in hadamard_matrix:
        sign_change = 0
        sign = row[0] > 0
        #   count sign changes
        for x in row:
            if sign == True:
                if x < 0:
                    sign_change += 1
                    sign = False
            else:
                if x > 0:
                    sign_change += 1
                    sign = True
            #   copy line to the index which equals the sign changes
        walsh_matrix[sign_change] = [x for x in row]
    return walsh_matrix
def create_nth_Haar(n):
    if n == 1:
        return H_sqrt2
    top_half = ut.kronecker_product(create_nth_Haar(n-1),[[1,1]])
    bottom_half = ut.kronecker_product(create_2_to_nth_I(n-1),[[1,-1]])
    matrix = np.vstack((top_half, bottom_half))
    return ut.kronecker_product([[1/sqrt_2]],matrix) # normalization

def create_nth_Haar_basis(n,delta):
    return ut.product(create_nth_Haar(n),create_nth_Standard_basis(n,delta))

def create_nth_Hadamard_basis(n,delta):
    return ut.product(create_nth_Hadamard(n),create_nth_Standard_basis(n,delta))

def create_nth_Walsh_Hadamard_basis(n,delta):
    return ut.product(create_nth_Walsh_Hadamard(create_nth_Hadamard(n)),create_nth_Standard_basis(n,delta))
def create_nth_Standard_basis(n,delta):
    partition = pow(2,n)
    return [[int(j == i)*sqrt(1/delta[i]) for j in range(partition)] for i in range(partition)]

def calculate_mse(energy, k_term_representation, k_terms, delta):
    mse = [energy - sum([(k_term_representation[term][i]**2)*delta[i] for i in range(len(k_term_representation[term]))]) for term in range(k_terms)]
    print(mse)
    return mse
def custom_permutation(original_list, permutation_indexes): # permutes a list based on a given permutation
    permuted_list = [original_list[i] for i in permutation_indexes]
    return permuted_list

def approximate(coefs, basis, delta):
    indexed_coefs = [(value, index) for index, value in enumerate(coefs)] # bind index and values together
    sorted_indexed_coefs = sorted(indexed_coefs,reverse=True, key=lambda x: x[0][0]**2) # sort by the power of l2 of the value
    sorted_coefs = [value for value, _ in sorted_indexed_coefs] # sorted reps
    sorted_indexes = [index for _, index in sorted_indexed_coefs] # permutation of the reps
    sorted_basis = custom_permutation(basis, sorted_indexes) # sorted base functions based of the permutaion of reps
    sorted_delta = custom_permutation(delta, sorted_indexes) # sorted delta_areas based of the permutaion of reps (relevant for different deltas)
    k_levels_approximation = []
    for i in range(len(sorted_indexed_coefs)):
        k_level_representation = [x for x in sorted_coefs] #  copy
        for j in range(i+1, len(sorted_indexed_coefs)):
            k_level_representation[j] = [0] #   erase the unneeded data
        k_levels_approximation.append(ut.inner_product(k_level_representation,sorted_basis)[0]) # calculate the approximation
    return [k_levels_approximation,sorted_delta]

def hw_function(x):
    return x*pow(math.e,x)  # given function
def hw_function_integral_over_area(left_x, right_x):
    return (right_x -1)*pow(math.e,right_x) - (left_x-1)*pow(math.e,left_x)     # integral from left x to right x of the given function using N-L

def hw_function_energy(left_x, right_x):
    y_left =  (pow(math.e, 2 * left_x)/2) * (pow(left_x,2) - left_x + 1/2)
    y_right =  (pow(math.e, 2 * right_x)/2) * (right_x**2 - right_x + 1/2)
    return y_right - y_left # integral from l_x to r_x of (function)^2

# # create bases for n= 2....6
# for n in range(2,7):
#     k_levels = pow(2,n)
#     left_bound, right_bound = 0, 1
#     delta_area = [(right_bound - left_bound)/k_levels for i in range(k_levels)]
#     x_values = np.linspace(left_bound, right_bound, k_levels + 1)
#     hadamard_basis = create_nth_Hadamard_basis(n,delta_area)
#     ut.plot_rows(x_values,hadamard_basis, "Hadamard/h" + str(n))
#     walsh_hadamard_basis = create_nth_Walsh_Hadamard_basis(n,delta_area)
#     ut.plot_rows(x_values,walsh_hadamard_basis, "Walsh_Hadamard/hw" + str(n))
#     haar_basis = create_nth_Haar_basis(n,delta_area)
#     ut.plot_rows(x_values,haar_basis, "Haar/ha" + str(n))

#set the parameters
n = 6
k_levels = pow(2,n)
left_bound, right_bound = -4, 5
delta_area = [(right_bound - left_bound) / k_levels for i in range(k_levels)]

# Create an array of x-values
x_values = np.linspace(left_bound, right_bound, k_levels+1)

# Evaluate the integral of the function at each interval
y_values =[]
for i in range(1,k_levels+1):
    y_values.append([hw_function_integral_over_area(x_values[i-1], x_values[i])])

#   calculate the functions energy
func_energy = hw_function_energy(x_values[0], x_values[-1])
terms_x_axis = [i for i in range(0,k_levels+1)]
#   standard basis
standard_basis = create_nth_Standard_basis(n, delta_area)
standard_coef = ut.product(standard_basis,y_values)
standard_approximation = approximate(standard_coef,standard_basis,delta_area)
ut.plot_rows( x_values,standard_approximation[0], "Standard/Standard Approximation")
ut.plot(terms_x_axis, calculate_mse(func_energy, standard_approximation[0], k_levels,standard_approximation[1]), "Standard/Standard_MSE_plot")

#   hadamard basis
hadamard_basis = create_nth_Hadamard_basis(n, delta_area)
hadamard_coef = ut.product(hadamard_basis, y_values)
hadamard_approximation = approximate(hadamard_coef,hadamard_basis,delta_area)
ut.plot_rows( x_values,hadamard_approximation[0], "Hadamard/Hadamard Approximation")
ut.plot(terms_x_axis, calculate_mse(func_energy, hadamard_approximation[0], k_levels,hadamard_approximation[1]), "Hadamard/Hadamard_MSE_plot")
#   walsh hadamard basis
walsh_hadamard_basis = create_nth_Walsh_Hadamard_basis(n, delta_area)
walsh_hadamard_coef = ut.product(walsh_hadamard_basis, y_values)
walsh_hadamard_approximation = approximate(walsh_hadamard_coef,walsh_hadamard_basis,delta_area)
ut.plot_rows( x_values, walsh_hadamard_approximation[0], "Walsh_Hadamard/Walsh Hadamard Approximation")
ut.plot(terms_x_axis, calculate_mse(func_energy, walsh_hadamard_approximation[0], k_levels,walsh_hadamard_approximation[1]), "Walsh_Hadamard/Walsh_Hadamard_MSE_plot")
#   haar basis
haar_basis = create_nth_Haar_basis(n, delta_area)
haar_coef = ut.product(haar_basis,y_values)
haar_approximation = approximate(haar_coef,haar_basis,delta_area)
ut.plot_rows( x_values,haar_approximation[0], "Haar/Haar Approximation")
ut.plot(terms_x_axis, calculate_mse(func_energy, haar_approximation[0], k_levels,haar_approximation[1]), "Haar/Haar_MSE_plot")
