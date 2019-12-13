"""
    This is a rough implementation, since it's based on WORDS composed with letters only. No email, phone number
    and any other features are considered yet.
"""

import numpy as np
import os
import re
import copy
import random as rand

# !!! Attention: paths must be changed if this program runs on different machines or data sets!
# Paths:
path_spam_train = 'C:/Users/User/Desktop/Computer Science/6375 machine learning/Assignments/Assignment_3/train/spam'
path_ham_train = 'C:/Users/User/Desktop/Computer Science/6375 machine learning/Assignments/Assignment_3/train/ham'
path_spam_test = 'C:/Users/User/Desktop/Computer Science/6375 machine learning/Assignments/Assignment_3/test/spam'
path_ham_test = 'C:/Users/User/Desktop/Computer Science/6375 machine learning/Assignments/Assignment_3/test/ham'

'''
    Global data:
'''

pattern1 = re.compile('[a-zA-Z]+')

'''
    Get unique words and their frequency in a dictionary. This method determines the dimension of this problem.
    Input: all files, pattern.
    Output: The huge text block.
'''


def file_loader(path0, path1):
    spam_files = os.listdir(path0)
    ham_files = os.listdir(path1)

    # Global block of text contents:
    text_block = ""

    for file in spam_files:
        text_block = text_block + ' ' + open(path0 + '/' + file, 'r', encoding='utf-8', errors='ignore').read()

    for file in ham_files:
        text_block = text_block + ' ' + open(path1 + '/' + file, 'r', encoding='utf-8', errors='ignore').read()

    return text_block


'''
    Count number of spam files and ham files.
    Input: paths
    Output: number of files.
'''


def count_files(path0, path1):
    spam_files = os.listdir(path0)
    ham_files = os.listdir(path1)
    return len(spam_files), len(ham_files)


'''
    Get unique words.
    Input: text chunk, pattern to match
    Output: unique words dictionary
'''


def unique_words(text, pattern):
    words = pattern.findall(text)
    unique = {}
    for word in words:
        if unique.get(word) is None:
            unique[word] = 0
    return unique


'''
    Load separate file from disk:
    Input: paths to spam files and ham files, pattern
    Output: [ [file1 words], [file2 words], ... [file10000 words] ] 
'''


def loader_single_file(path0, path1, pattern):
    spam_files = os.listdir(path0)
    ham_files = os.listdir(path1)

    file_word = []  # Contain all filtered words from each file
    for file in spam_files:
        txt = open(path0 + '/' + file, 'r', encoding='utf-8', errors='ignore').read()
        words = pattern.findall(txt)
        file_word.append(words)

    for file in ham_files:
        txt = open(path1 + '/' + file, 'r', encoding='utf-8', errors='ignore').read()
        words = pattern.findall(txt)
        file_word.append(words)

    return np.asarray(file_word)


'''
    Get an numpy matrix of data points (files) x coordinates.
    Input: [[file1 words], [file2 words], ... ,[file10000 words]], weight vector to know the dimension of this problem
    Output: Matrix that: 
        Rows = number of data points; 
        Columns = dimension of this problem + 1
'''


def coord_mat(w, words_each_file):
    co_x = np.ndarray(shape=(len(words_each_file), len(w)))     # spam_x: All data points x coordinates
    co_x[:, 0] = 1

    i = 0
    j = 1
    for word_block in words_each_file:
        file_dict = unique_dict
        for word in word_block:                                    # Generate a dictionary with frequency in it
            if file_dict.get(word) is not None:
                val = file_dict[word] + 1
                file_dict[word] = val
        for key in file_dict:                                      # Generate x coordinates by dictionary
            co_x[i, j] = file_dict[key]
            j += 1
        j = 1
        i += 1
        for key in unique_dict:  # Reset unique_dict
            unique_dict[key] = 0
    return co_x


''' 
    Deprecated, since it's slower than adv_l_w, and higher CPU occupation.
    Calculate l(w) for one weight by iterating through all files, each file as a data point.
    Input: weight vector, coordinate matrix, tuple of numbers of files, index of which weight is using
    Output: l(w)
'''


def l_w(w, mat_of_coord, classes, index):
    dots = np.dot(mat_of_coord, w)                              # used to calculate sigmoid
    dots = np.where(dots > 36, 36, dots)                        # Prevent overflow when calculating exp(dots)
    y_l = np.zeros((classes[0] + classes[1], 1))
    y_l[classes[0]:classes[1] + classes[0], 0] = 1               # yl
    sigmoid = np.exp(dots)/(1 + np.exp(dots))                   # sigmoid
    x_i = mat_of_coord[:, index]
    x_i = np.array([x_i]).T                                     # xi
    lw_l = np.multiply(x_i, (np.nan_to_num(y_l - sigmoid)))     # Deal with NaN values
    lw = np.sum(lw_l)
    return lw


'''
    Testing test files, discriminate them.
    Input: test cases' coordinates, weights
    Output: a vector with P(Y = 1|X, W) = 1 and P(Y = 0|X, W) = 0
'''


def discriminate(w, test_coord_mat):
    dots = np.dot(test_coord_mat, w)  # used to calculate sigmoid
    dots = np.where(dots > 36, 36, dots)  # Prevent overflow when calculating exp(dots)
    sigmoid = np.exp(dots) / (1 + np.exp(dots))  # sigmoid
    result = np.where(sigmoid > 0.5, 1, 0)
    return result


'''
    Get dots of weights and coordinates.
    Input: weight vector, coordinates, index of weight are updating.
    Output: dots for all data to calculate sigmoid, the part of np.dot(weights, mat_coord) needs update.
'''


def get_dots(w, mat_of_coord, index):
    old_dots = np.dot(mat_of_coord, w)
    up_coord = mat_of_coord[:, index]
    return old_dots, up_coord


'''
    Calculate l(w) for one weight by iterating through all files, each file as a data point.
    Input: dots for sigmoid, coordinate matrix, tuple of numbers of files, index of which weight is using
    Output: l(w)
'''


def adv_l_w(dots, mat_of_coord, classes, index):
    dots = np.where(dots > 36, 36, dots)                        # Prevent overflow when calculating exp(dots)
    y_l = np.zeros((classes[0] + classes[1], 1))
    y_l[classes[0]:classes[1] + classes[0], 0] = 1               # yl
    sigmoid = np.exp(dots)/(1 + np.exp(dots))                   # sigmoid
    x_i = mat_of_coord[:, index]
    x_i = np.array([x_i]).T                                     # xi
    lw_l = np.multiply(x_i, (np.nan_to_num(y_l - sigmoid)))     # Deal with NaN values
    lw = np.sum(lw_l)
    return lw


# Get dimension of the problem
chunk = file_loader(path_spam_train, path_ham_train)            # Get chunk
num_tup = count_files(path_spam_train, path_ham_train)          # Get number of files
unique_dict = unique_words(chunk, pattern1)                 # Get unique words
dim = len(unique_dict)                                  # The dimension

# Load files' data from disk:
file_words = loader_single_file(path_spam_train, path_ham_train, pattern1)

# Arbitrary initial weight vector:
weights = []
for i in range(dim + 1):  # Initial weight vector arbitrarily
    weights.append(0.5)
weights = np.asarray(weights)  # Turn into numpy array
weights = np.array([weights]).T

# Load data points x coordinates:
data_mat = coord_mat(weights, file_words)

# Scale data_mat:


# Loop to calculate weight vector:
n_long = 0.01
lamb_da = 10

for i in range(dim + 1):
    o_dots, up_col = get_dots(weights, data_mat, i)
    up_col = np.array([up_col]).T

    w_old = 0
    w_new = 1                                       # Arbitrary number, just make sure (10 - 0.5) is large
    while np.abs(w_new - w_old) > 0.001:
        w_old = copy.deepcopy(weights[i])
        rd = (rand.random() - 0.9)/100               # Small turbulence to prevent from stuck when descending
        weights[i] = \
            weights[i] + (n_long + rd) * (adv_l_w(o_dots, data_mat, num_tup, i)) - lamb_da * (n_long + rd) * weights[i]
        w_new = weights[i]
        o_dots = o_dots + (up_col * (w_new - w_old))
    print('Calculating %s weight: ' % i)

# Classify test cases:
num_test_spam, num_test_ham = count_files(path_spam_test, path_ham_test)

tests_file_coord = loader_single_file(path_spam_test, path_ham_test, pattern1)
test_spam_coord = tests_file_coord[0: num_test_spam]
test_ham_coord = tests_file_coord[num_test_spam:]

test_spam_mat = coord_mat(weights, test_spam_coord)
test_ham_mat = coord_mat(weights, test_ham_coord)

cond_p_spam_test = discriminate(weights, test_spam_mat)
spam_as_ham = np.sum(cond_p_spam_test)
spam_as_spam = len(cond_p_spam_test) - spam_as_ham
correct_ratio_on_spam = spam_as_spam / len(cond_p_spam_test)

cond_p_ham_test = discriminate(weights, test_ham_mat)
ham_as_ham = np.sum(cond_p_ham_test)
ham_as_spam = len(cond_p_ham_test) - ham_as_ham
correct_ratio_on_ham = ham_as_ham / len(cond_p_ham_test)


print('Test on spam set: ')
print('Total: ', len(cond_p_spam_test))
print('Spam: ', spam_as_spam)
print('Ham: ', spam_as_ham)
print('Accuracy: ', correct_ratio_on_spam)

print('Test on ham set: ')
print('Total: ', len(cond_p_ham_test))
print('Ham: ', ham_as_ham)
print('Spam: ', ham_as_spam)
print('Accuracy: ', correct_ratio_on_ham)
