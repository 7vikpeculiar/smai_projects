import numpy as np
from PIL import Image
import PIL
import os
import sys
import math

def take_input():
    input_data = open(sys.argv[1]).readlines()
    input_dict = []
    for i in range(len(input_data)):
        ele = input_data[i].split(" ")
        if ele[1][-1] == "\n":
            ele[1] = ele[1][:-1]
        input_dict.append([ele[0],ele[1]])
    return input_dict

def take_output():
    output_data = open(sys.argv[2]).readlines()
    output_dict = []
    for i in range(len(output_data)):
        ele = output_data[i]
        if ele[-1] == '\n':
            ele = ele[:-1]
        output_dict.append([ele," "])
    return output_dict

def find_X(imlist):
    start_arr = np.array(Image.open(imlist[0][0]), 'f').flatten()
    for imname in imlist[1:]:
      try:
          tmp_arr = np.array(Image.open(imname[0])).flatten()
          start_arr = np.vstack((start_arr, tmp_arr))
      except:
          print(imname + ' --> has been skipped while finding X matrix')
    return start_arr

def pca(imlist):
    X = find_X(imlist)
    no_img, dim = X.shape
    mean_X = X.mean(axis = 0)
    X = X - mean_X
    Xtr = np.transpose(X)
    if dim > no_img:
        M  = np.dot(X,Xtr)
        e, EV = np.linalg.eigh(M)
        EV = np.transpose(EV)
        tmp = np.transpose(np.dot(Xtr, EV))
        V = tmp[::-1]
        e = e[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= np.linalg.norm(V[:,i])
    else:
        U,S,V = np.linalg.svd(X)
        V = V[:num_data]
    return X, V,e,mean_X

def find_new_comps(X,V,e,immean,num_comps):
    Vt = np.transpose(V)
    C = np.matmul(X,Vt)
    c1, c2  = C.shape
    shortened = C[:,:num_comps]
    return shortened

def list_classes(imlist):
    class_dict = {}
    for ele in range(len(imlist)):
        piece = imlist[ele]
        if piece[1] not in class_dict:
            class_dict[piece[1]] = []
        class_dict[piece[1]].append(ele)
    return class_dict

def find_single_stat(list,C):
    l = C[list]
    mean_l = l.mean(axis=0)
    resurfed = l - mean_l
    resurfaced = np.power(resurfed,2)
    cov  = resurfaced.mean(axis=0)
    return mean_l, cov

def find_stats(class_list, shortened_C):
    key_list = class_list.keys()
    m_dict = {}
    cov_dict = {}
    for ele in key_list:
        mn,cv = find_single_stat(class_list[ele],C)
        m_dict[ele] = mn
        cov_dict[ele] = cv
    return m_dict,cov_dict

#--- TRAINING ---

training_data = take_input()
X,V,S,immean = pca(training_data)
C  = find_new_comps(X,V,S,immean,num_comps=32)
class_list = list_classes(training_data)
m,cov = find_stats(class_list, C)
#print(m,cov)

#--- TESTING ---
def find_test_X(imlist):
    start_arr = np.array(Image.open(imlist[0][0]), 'f').flatten()
    for imname in imlist[1:]:
      try:
          tmp_arr = np.array(Image.open(imname[0])).flatten()
          start_arr = np.vstack((start_arr, tmp_arr))
      except:
          print(imname + ' --> has been skipped while finding X matrix')
    return start_arr

def find_class(imlist):
     X = find_test_X(imlist)
     return X

def guassian(x, m,v):
    p1 = v**-0.5
    p2 = ((x-m)/v)**2
    val =  p1*math.exp(-p2)
    return val

def find_prob(x,mu,cov):
    prob = 1
    for i in range(32):
        prob *= guassian(x[i],mu[i],cov[i])
    return prob

def determine_class(x,mu,cov):
    lst = mu.keys()
    prob_dict = {}
    for ele in lst:
        prob_dict[ele] = find_prob(x,mu[ele],cov[ele])
    key_max = max(prob_dict.keys(), key=(lambda k: prob_dict[k]))
    return key_max

testing_data = take_output()
shp = find_test_X(testing_data)
Vt = np.transpose(V)
i = 0
for ele in shp:
    a = np.dot(ele,Vt)
    print(testing_data[i][0],determine_class(a,m,cov))
    i+=1

"""
m1,m2 = immean.shape
mean_phot = immean.flatten()
reconstr = np.matmul(C[:,:num_comps],V[:num_comps])
return reconstr

out=[]
out2 = []
for i in range(100):
    print(i)
    reconstr = np.matmul(C[:,:i],V[:i])
    arr = np.linalg.norm(reconstr)
    out.append(arr/float(10**13))
    out2.append(i)
    #print("Co",reconstr.shape)
print(out)
plt.scatter(out,np.zeros(len(out)))
plt.show()
"""

"""
print(X.shape, V.shape, S.shape, immean.shape)

#m,n = np.array(Image.open(training_data[0])).shape
#no_img = len(training_data)
#X,V,S,immean = pca(training_data)
"""
