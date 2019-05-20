import numpy as np
import math
import csv
import timeit


def make_data():
    org = np.random.randint(5000, size=(100, 2))
    new = np.random.randint(5000, size=(100, 2))
    arr = []
    # ranges = []
    for i,j in enumerate(org):
        values = np.linalg.norm(new-j, axis=1)
        arr.append(values)
    # print(arr)  
    # print(ranges)
    arr = np.array(arr)
    return arr

def loop(data, mean, arr):
    for i in range(len(arr)):
        # print("i: ", i)
        total = 0
        locs = []
        # print(data[0])
        for k, j in enumerate(data[0]):
            # print("j: ", j)
            if j == i:
                total += 1
                locs.append(k)
        if total > 1:
            best_val = -1
            # print(locs)
            for l in locs:
                val = abs(mean[l]-arr[l][i])
                # print(val)
                if val > best_val:
                    best_val = abs(mean[l]-arr[l][i])
                    best = l
                # print("best: ", best)
            for n, m in enumerate(data[0]):
                if n in locs and not (n == best):
                    data[0][n] = i+1
        # print("data[0]: ", data[0])
    return data

def new_sort():
    org = np.random.randint(5000, size=(CONST, 2))
    new = np.random.randint(5000, size=(CONST, 2))
    arr = []
    # ranges = []
    for i,j in enumerate(org):
        values = np.linalg.norm(new-j, axis=1)
        arr.append(values)
    # print(arr)  
    # print(ranges)
    arr = np.array(arr)
    sorted_arg_mins = []
    for i in arr:
        sorted_arg_mins.append(np.argsort(i, kind='mergesort'))
    # print(sorted_arg_mins)
    data = np.transpose(sorted_arg_mins)
    # print(data)
    mean = np.mean(arr, axis=1)
    count = 0
    while (not (len(np.unique(data[0])) == len(data[0])) or np.max(data[0]) >= len(data[0])):
        data = loop(data, mean, arr)
        data = np.where(data >= len(data[0]), 0, data)
        # print("data: ",data)
    # print(data[0])
    return data[0] 
        
    
            

def exhaustive(arr):
    arr = np.ravel(arr)
    combs = [
        [0,4,8],
        [0,5,7],
        [1,5,6],
        [1,3,8],
        [2,4,6],
        [2,3,7]        
    ]
    best_sum = 50000000
    for j,i in enumerate(combs):
        my_sum = arr[i[0]]+arr[i[1]]+arr[i[2]] 
        if my_sum < best_sum:
            best_sum = my_sum
            best = j
    # print(best_sum)
    # print("exhaustive: ", combs[best])
    return combs[best]
                   
             
               

def my_sort_2():
    org = np.random.normal(10, 2, size=(4, 2))
    new = np.random.normal(10, 2, size=(4, 2))
    arr = []
    # ranges = []
    for i,j in enumerate(org):
        values = np.linalg.norm(new-j, axis=1)
        arr.append(values)
    # print(arr)  
    # print(ranges)
    arr = np.array(arr)
    ranges = np.std(arr, axis=1)
    print(ranges)
    best, vals = exhaustive(arr)
    x_range = np.argsort(ranges, kind='mergesort')
    # print(x_range)
    # print(np.argmax(x_range))
    results = np.empty_like(new)
    best_sum = 0
    contained1 = 0
    for i in np.flip(x_range):
        row = i
        col = np.argmin(arr[row])
        best_sum += arr[row][col]
        results[row] = new[col]
        arr[:, col] = 500000000
        # print(arr)
        val = col+row*4
        if val in vals:
            contained1 += 1
        print(col+row*4)
    print(best_sum)
    print(best)
    return best - best_sum, contained1
    # print(results)


def my_sort():
    org = np.random.normal(1000, 500, size=(4, 2))
    new = np.random.normal(1000, 500, size=(4, 2))
    arr = []
    ranges = []
    for i,j in enumerate(org):
        values = np.linalg.norm(new-j, axis=1)
        ranges.append(np.max(values)-np.min(values))
        arr.append(values)
    # print(arr)  
    # print(ranges)
    arr = np.array(arr)
    best, vals = exhaustive(arr)
    x_range = np.argsort(ranges, kind='mergesort')
    # print(x_range)
    # print(np.argmax(x_range))
    results = np.empty_like(new)
    best_sum = 0
    contained = 0
    for i in np.flip(x_range):
        row = i
        col = np.argmin(arr[row])
        best_sum += arr[row][col]
        results[row] = new[col]
        arr[:, col] = 500000000
        # print(arr)
        val = col+row*4
        if val in vals:
            contained+=1
        # print(col+row*4)
    # print(best_sum)
    return best - best_sum, contained
    # print(results)


with open('test_data_matrix10000.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    data_all = []
    data = list(range(1,21))
    data_all.append(data)
    print(data)
    for _ in range(3):
        data = []
        for i in range(20):
            CONST = (i+1)*10
            print(CONST)
            data.append(timeit.timeit("new_sort()", setup="from __main__ import new_sort",number=10000))
        data_all.append(data)
    print(data_all)
    writer.writerows(data_all)
# total = 0
# for i in range(10000):
#     print(i)
#     arr = make_data()
#     data = new_sort(arr)
#     for i,j in enumerate(data):
#         data[i] = i*len(data)+j
#     # print(data)
#     ex = exhaustive(arr)
#     j = (ex == data)
#     if j.all():
#         total += 1
# print(total/10000)
# correct = 0
# close = 0
# h = 0
# correct_values = 0
# for _ in range(1):    
#     result, contained = my_sort_2()
#     correct_values += contained
#     if result == 0:
#         correct+=1
#     if abs(result)<100:
#         h+=1
#     close += abs(result)
# print(correct) #/100000*100
# # print(h/100000*100)
# # print(close/100000)
# print(correct_values)
# # for i in range(20):
#     # CONST = (i+1)*10
#     # print(CONST)
#     # 