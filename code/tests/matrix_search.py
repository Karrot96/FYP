import numpy as np
import math
import timeit

def my_sort():
    org = np.random.randint(2000, size=(50,2))
    new = np.random.randint(2000, size=(50,2))
    arr = []
    for i,j in enumerate(org):
        values = np.linalg.norm(new-j, axis=1)
        arr.append(values)    
    len_x = len(arr)
    arr = np.ravel(arr)
    index = np.argsort(arr, kind='mergesort')
    values = np.sort(arr, kind='mergesort')
    result = np.empty_like(org)
    x_done = []
    y_done = []
    for k, n in enumerate(values):
        for i,j in enumerate(index):
            # print(j,k)
            if j == k:
                new_value = i%len_x
                if new_value not in x_done:
                    old_value = int(i/len_x)
                    if old_value not in y_done:
                        result[old_value] = [new_value]
                        y_done.append(old_value)
                        x_done.append(new_value)
                if len(x_done) == len(org):
                    # print(org)
                    # print(result)
                    return 0

print(timeit.timeit("my_sort()", setup="from __main__ import my_sort",number=100))