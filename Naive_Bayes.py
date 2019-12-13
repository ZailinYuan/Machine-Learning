"""
    This is a rough implementation, since it's based on WORDS composed with letters only. No email, phone number
    and any other features are considered yet.
"""


import math
import os
import re

'''
    Global variables:
'''
# !!! Attention: paths must be changed if this program runs on different machines or data sets!
# Paths:
path_spam_train = 'train/spam'
path_ham_train = 'train/ham'
path_spam_test = 'test/spam'
path_ham_test = 'test/ham'

# text concatenate:
text_spam_train = ""
text_ham_train = ""
text_spam_test = ""
text_ham_test = ""

# number of files:
num_spam_file_train = 0
num_ham_file_train = 0
num_spam_file_test = 0
num_ham_file_test = 0

# text block size:
size_spam_text_train = 0
size_ham_text_train = 0
size_spam_text_test = 0
size_ham_text_test = 0

# Patterns:
pattern1 = re.compile('[a-zA-Z]+')

'''
    Given a directory, store number of files and concatenate contents of these files in global variables:
    Input: source file directory
    Output: text block (concatenate of texts of all files under that directory)
'''


def file_loader(path):
    files = os.listdir(path)
    # Return info:
    file_num = 0
    text_block = ""
    for file in files:
        file_num = file_num + 1
        each_file = open(path + '/' + file, 'r', encoding='utf-8', errors='ignore')
        text_block = text_block + ' ' + each_file.read()
    return text_block, file_num


'''
    Given text, return words frequency of it in a dictionary:
    Input: block of text
    Output: word frequency table, total number of terms in text block
'''


def words_frequency(texts, pattern):
    # All terms in block:
    terms = pattern.findall(texts)

    # Get term frequency table:
    term_fre = {}
    sum_terms = len(terms)
    for term in terms:
        if term_fre.get(term) is not None:
            counter = term_fre[term]
            counter = counter + 1
            term_fre[term] = counter
        else:
            term_fre[term] = 1
    return term_fre, sum_terms


'''
    Get word probability of showing up:
    Input: word frequency table
    Output: word probability table
'''


def get_probability(term_fre, total):
    term_p = {}
    for term in term_fre:
        fre = term_fre[term]
        p = fre / total
        term_p[term] = p
    return term_p


'''
    Get Posterior p:
    Input: term frequency table, total number of terms
    Output: Laplace smoothed posterior probability
'''


def conditional_p(term_fre, total):
    # b is total number of unique term:
    b = len(term_fre)

    # base is sum of frequency + number of unique terms, base for Laplace smooth:
    base = b + total

    # posterior table (term - cond_p):
    posterior_laplace = {}
    for term in term_fre:
        cond_p_term = ((term_fre[term] + 1) / base)
        posterior_laplace[term] = cond_p_term

    return posterior_laplace


'''
    Get posterior.
    Input: prior probability, test cases' frequency table, conditional probability from training
    Output: posterior probability for each test case    
'''


def get_posterior(prior, fre_table, cond_p, train_fre):
    post = 0

    # Get base:
    total_unique = len(train_fre)
    term_total = 0
    for each in train_fre:
        term_total = term_total + train_fre[each]
    base = total_unique + term_total

    # First term in log equation of posterior:
    log_prior = math.log(prior, 2)

    # Second term in log equation of posterior:
    sum_log_cond_p = 0
    for each in fre_table:
        fre = fre_table[each]
        cond = 0
        if cond_p.get(each) is not None:
            cond = cond_p[each]
        else:
            cond = (1 / base)

        log_cond = math.log(cond, 2)
        sum_log_cond_p = sum_log_cond_p + fre * log_cond

    # return posterior:
    post = log_prior + sum_log_cond_p
    return post


# Get spam training text block / file amount:
block = file_loader(path_spam_train)
text_spam_train = block[0]
num_spam_file_train = block[1]

# Get ham training text block / file amount:
block = file_loader(path_ham_train)
text_ham_train = block[0]
num_ham_file_train = block[1]

#############################################################################

# Get spam word frequency and total terms:
text_block_stat = words_frequency(text_spam_train, pattern1)
spam_fre_table = text_block_stat[0]
size_spam_text_train = text_block_stat[1]

# Get ham word frequency and total terms:
text_block_stat = words_frequency(text_ham_train, pattern1)
ham_fre_table = text_block_stat[0]
size_ham_text_train = text_block_stat[1]

###############################################################################

# Get Laplace smoothed conditional probability table:
# cond_p_spam: term - P(term | C)
cond_p_spam = conditional_p(spam_fre_table, size_spam_text_train)
cond_p_ham = conditional_p(ham_fre_table, size_ham_text_train)

# Get prior probabilities:
# prior_spam: a number
prior_spam = (num_spam_file_train / (num_spam_file_train + num_ham_file_train))
prior_ham = (num_ham_file_train / (num_spam_file_train + num_ham_file_train))

###########################################################################

# Start test:

# Test on spam files:
spam_files = os.listdir(path_spam_test)
detect_as_spam = []
for spam in spam_files:
    each_spam = open(path_spam_test + '/' + spam, 'r', encoding='utf-8', errors='ignore')
    spam_txt = each_spam.read()
    terms_spam = pattern1.findall(spam_txt)

    # frequency table for each spam file:
    test_spam_fre = {}
    for each in terms_spam:
        if test_spam_fre.get(each) is not None:
            v = test_spam_fre[each]
            test_spam_fre[each] = (v + 1)
        else:
            test_spam_fre[each] = 1

    # posterior for each file:
    spam_is_spam_p = get_posterior(prior_spam, test_spam_fre, cond_p_spam, spam_fre_table)
    spam_is_ham_p = get_posterior(prior_ham, test_spam_fre, cond_p_ham, ham_fre_table)

    # detect results:
    if spam_is_spam_p > spam_is_ham_p:
        detect_as_spam.append(1)
    else:
        detect_as_spam.append(0)

spam_succ = 0
spam_fail = 0
for result in detect_as_spam:
    if result == 1:
        spam_succ = spam_succ + 1
    else:
        spam_fail = spam_fail + 1

print('Test on spam files:')
print('Spam detected as spam: ', spam_succ)
print('Spam detected as ham: ', spam_fail)
print('Accuracy: ', (spam_succ / (spam_succ + spam_fail)))


# Test on ham files:
ham_files = os.listdir(path_ham_test)
detect_as_ham = []
for ham in ham_files:
    each_ham = open(path_ham_test + '/' + ham, 'r', encoding='utf-8', errors='ignore')
    ham_txt = each_ham.read()
    terms_ham = pattern1.findall(ham_txt)

    # frequency table for each ham file:
    test_ham_fre = {}
    for each in terms_ham:
        if test_ham_fre.get(each) is not None:
            v = test_ham_fre[each]
            test_ham_fre[each] = (v + 1)
        else:
            test_ham_fre[each] = 1

    # posterior for each file:
    ham_is_ham_p = get_posterior(prior_ham, test_ham_fre, cond_p_ham, ham_fre_table)
    ham_is_spam_p = get_posterior(prior_spam, test_ham_fre, cond_p_ham, spam_fre_table)

    # detect results:
    if ham_is_ham_p < ham_is_spam_p:
        detect_as_ham.append(1)
    else:
        detect_as_ham.append(0)

ham_succ = 0
ham_fail = 0
for result in detect_as_ham:
    if result == 1:
        ham_succ = ham_succ + 1
    else:
        ham_fail = ham_fail + 1

print('Test on ham files:')
print('Ham detected as ham: ', ham_succ)
print('Ham detected as spam: ', ham_fail)
print('Accuracy: ', ham_succ / (ham_succ + ham_fail))














