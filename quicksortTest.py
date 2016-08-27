"""
Learning how to use SCIKIT-LEARN

Digital Gym Research Group
"""

# def quicksort(arr):
#     try:
#         len(arr)
#     except:
#         print(arr)

#     if len(arr) <= 1:
#         return arr
#     else:
#         l, pivot, r = partition(arr)
#         return quicksort(l)+[pivot]+quicksort(r)

# def partition(arr):
#     pivot, arr = arr[len(arr)-1], arr[:len(arr)-1]
#     l = [x for x in arr if x <= pivot]
#     r = [x for x in arr if x > pivot]
#     return l, pivot, r




def quicksort(tarr):
    """
    Input must be a enumerated list with the index in [0] and value in [1]
    """
    if len(tarr) <= 1:
        return tarr
    else:
        l, pivot, r = partition(tarr)
        return quicksort(l)+[pivot]+quicksort(r)

def partition(tarr):
    pivot, tarr = tarr[len(tarr)-1], tarr[:len(tarr)-1]
    l = [x for x in tarr if x[1] <= pivot[1]]
    r = [x for x in tarr if x[1] > pivot[1]]
    return l, pivot, r

print(len([(0,4),(1,10)]))
sample = list(enumerate([1,2,3,4,6,2]))
print(sample)
obj = quicksort(sample)


print(obj)