import random

def bubble_sort(numbers):

    n = len(numbers)
    for i in range(n):
        for j in range(0, n - i - 1):
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
    return numbers


def merge_sort(numbers):
    """
    Sortuje liczby metodą merge sort (sortowanie przez scalanie).
    """
    if len(numbers) > 1:
        # Dzielenie listy na dwie połowy
        mid = len(numbers) // 2
        left_half = numbers[:mid]
        right_half = numbers[mid:]

        # Rekurencyjne sortowanie obu połówek
        merge_sort(left_half)
        merge_sort(right_half)

        # Scalanie posortowanych połówek
        i = j = k = 0

        # Porównywanie i łączenie
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                numbers[k] = left_half[i]
                i += 1
            else:
                numbers[k] = right_half[j]
                j += 1
            k += 1

        # Dodawanie pozostałych elementów z lewej połowy
        while i < len(left_half):
            numbers[k] = left_half[i]
            i += 1
            k += 1

        # Dodawanie pozostałych elementów z prawej połowy
        while j < len(right_half):
            numbers[k] = right_half[j]
            j += 1
            k += 1

    return numbers


N = 20
random_numbers = [random.randint(1, 100) for _ in range(N)]

bubble_sorted = bubble_sort(random_numbers.copy())
insertion_sorted = merge_sort(random_numbers.copy())

correct_sorted = sorted(random_numbers)

# Wyniki
import pandas as pd

results = pd.DataFrame({
    "Oryginalne liczby": random_numbers,
    "Bubble sort": bubble_sorted,
    "Merge sort": insertion_sorted,
    "Wbudowane sortowanie": correct_sorted
})

print(results)
