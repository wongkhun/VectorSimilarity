import math 
import numpy as np
from scipy.spatial.distance import cdist, pdist

# https://cloud.tencent.com/developer/article/1668762
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

def inner_product(q,d):
    return sum([(a-b)**2 for (a,b) in zip(q,d)])

def euclidean_distance(q,d):
    return math.sqrt(sum([(a-b)**2 for (a,b) in zip(q,d)]))

def manhattan_distance(q,d):
    return sum([abs(a-b) for (a,b) in zip(q,d)])

def chebyshev_distance(q,d):
    return max([abs(a-b) for (a,b) in zip(q,d)])

def minkowski_distance(q,d,p=1):
    return math.sqrt(sum([(a-b)**p for (a,b) in zip(q,d)]))

def standard_euclidean_distance(q,d):
    q = np.asarray(q)
    d = np.asarray(d)
    X = np.vstack([q,d])
    sk = np.var(X, axis=0, ddof=1)
    return np.sqrt(((q-d)**2 / sk).sum())

def mahalanobis_distance(q,d):
    q = np.asarray(q)
    d = np.asarray(d)
    X = np.vstack([q,d])
    XT= X.T
    S = np.cov(X)
    SI = np.linalg.inv(S)
    n = XT.shape[0]
    distance = [] 
    for i in range(0, n):
        for j in range(i+1, n):
            delta = XT[i] - XT[j]
            d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            distance.append(d)
    return distance

def lance_williams_distance(q,d):
    distance = 0
    for a,b in zip(q,d):
        step_distance = abs(a-b) / (a+b)
        distance += step_distance
    return distance

def cosine_similarity(q, d):
    sum_qd = 0.0
    norm_q = 0.0 
    norm_d = 0.0 
    for a,b in zip(q,d):
        sum_qd += a * b 
        norm_q += a ** 2 
        norm_d += b ** 2 
    return sum_qd / ((norm_q * norm_d) ** 0.5)
    
def tanimoto_coefficient_relation(q,d):
    sum_qd = 0.0
    norm_q = 0.0 
    norm_d = 0.0 
    for a,b in zip(q,d):
        sum_qd += a * b 
        norm_q += a ** 2 
        norm_d += b ** 2 
    return sum_qd / (norm_q + norm_d - sum_qd)

def pearson_correlation(q,d):
    return np.corrcoef(q,d)[0][1]

def jaccard_similarity(q,d):
    unions = len(set(q).union(set(d)))
    intersections = len(set(q).intersection(set(d)))
    return intersections / unions

def jaccard_distance(q,d):
    return 1 - jaccard_similarity(q,d)




q = [0.1, 0.2, 0.4]
d = [0.12, 0.23, 0.2]
print(inner_product(q,d))
print(euclidean_distance(q,d))
print(manhattan_distance(q,d))
print(chebyshev_distance(q,d))
print(minkowski_distance(q,d))
print(standard_euclidean_distance(q,d))
print(mahalanobis_distance(q,d))
print(lance_williams_distance(q,d))
print(cosine_similarity(q,d))
print(tanimoto_coefficient_relation(q,d))
print(pearson_correlation(q,d))
print(jaccard_distance(q,d))
print(jaccard_similarity(q,d))