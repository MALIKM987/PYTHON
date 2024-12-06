import multiprocessing
import matplotlib.pyplot as plt
import random
import time
def sortit(data):
    return sorted(data)

def merge(left, right):
    i, j, k = 0, 0, 0
    ans = [0 for _ in range(len(left) + len(right))]
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            ans[k] = left[i]
            i+=1
        else:
            ans[k] = right[j]
            j += 1
        k+=1
    while i < len(left):
        ans[k] = left[i]
        i += 1
        k+=1
    while j < len(right):
        ans[k] = right[j]
        j += 1
        k+=1
    return ans


def parallel_sort(data, num_proc):
    if len(data) <= 1:
        return data

    div_len = (len(data) + num_proc - 1) // num_proc
    divisions = []
    for i in range(num_proc - 1):
        divisions.append(data[i * div_len: (i + 1) * div_len])
    if (num_proc - 1) * div_len < len(data):
        divisions.append(data[(num_proc - 1) * div_len:])

    with multiprocessing.Pool(processes=num_proc) as pool:
        sorted_divisions = pool.map(sortit, divisions)

    ans = []

    for i in range(len(sorted_divisions)):
        ans = merge(ans, sorted_divisions[i])

    return ans


if __name__ == "__main__":

    data = [random.randint(0, 1000) for _ in range(100)]
    print("Przed sortowaniem (pierwsze 100):", data[:100])
    sorted_data = parallel_sort(data, 6)
    print("Po sortowaniu (pierwsze 100):", sorted_data[:100])
