import multiprocessing

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def sequential_merge_sort(data):
    if len(data) <= 1:
        return data

    mid = len(data) // 2
    left = sequential_merge_sort(data[:mid])
    right = sequential_merge_sort(data[mid:])
    return merge(left, right)


def parallel_merge_sort(data):
    if len(data) <= 1:
        return data

    mid = len(data) // 2
    left = data[:mid]
    right = data[mid:]

    with multiprocessing.Pool(processes=2) as pool:
        sorted_parts = pool.map(parallel_merge_sort, [left, right])

    return merge(*sorted_parts)


if __name__ == "__main__":
    import random
    data = [random.randint(0, 1000) for _ in range(10)]
    print("Przed sortowaniem (pierwsze 10):", data[:10])
    sorted_data = parallel_merge_sort(data)
    print("Po sortowaniu (pierwsze 10):", sorted_data[:10])
