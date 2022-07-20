import numpy as np

arr = np.array([1,2,3,4,5,6,7,8,9 , 10 , 9,8,7,6,5,4,3,2,1,2 , 3 , 2,1,2,3,4,5,6,7 , 8 , 7,6,5,4,2,1])

thres = 7

prev_higher = np.where(arr[1:-1]>arr[2:],arr[1:-1],0)
next_higher = np.where(arr[1:-1]>arr[:-2],arr[1:-1],0)
threshold = np.where(arr[1:-1]>thres, arr[1:-1],0)

print(prev_higher)
print(next_higher)
print(threshold)

ploc = threshold * prev_higher * next_higher
print(ploc)

print(ploc.nonzero()) #Location of non zero value
print("Final Location =",ploc.nonzero()[0]+1) #To compensate for the first element which was not taken for spectral peak analysis

# Peak with value 3 not taken as threshold is 7