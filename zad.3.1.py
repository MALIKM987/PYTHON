import random
def get_matrix_input(rows, cols, matrix_name):
    matrix = []
    print(f"\nWprowadzanie wartości dla macierzy {matrix_name}:")
    for i in range(rows):
        row = []
        for j in range(cols):
            value = float(input(f"Podaj wartość dla {matrix_name}[{i + 1}][{j + 1}]: "))
            row.append(value)
        matrix.append(row)
    return matrix
def generate_random_matrix(rows, cols, matrix_name):

    matrix = []
    print(f"\nGenerowanie losowych wartości dla macierzy {matrix_name}...")
    for i in range(rows):
        row = [random.uniform(1, 100) for _ in range(cols)]  
        matrix.append(row)
    return matrix
def print_matrix(matrix, name):

    print(f"\n{name}:")
    for row in matrix:
        print(" ".join(f"{value:.2f}" for value in row))
def multiply_matrices(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        print("Macierze mają niewłaściwe wymiary do mnożenia!")
        return None
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result
def main():
    for matrix_num in [1, 2]:
        print(f"\nTworzenie macierzy {matrix_num}:")
        rows = int(input("Podaj liczbę wierszy: "))
        cols = int(input("Podaj liczbę kolumn: "))

        method = input("Czy chcesz generować wartości losowo? (tak/nie): ").strip().lower()
        if method == "tak":
            locals()[f"matrix_{matrix_num}"] = generate_random_matrix(rows, cols, f"Macierz {matrix_num}")
        else:
            locals()[f"matrix_{matrix_num}"] = get_matrix_input(rows, cols, f"Macierz {matrix_num}")
    result = multiply_matrices(locals()["matrix_1"],locals()["matrix_2"])
    print_matrix(result, "Macierz wynikowa")
    print_matrix(locals()["matrix_1"], "Macierz 1")
    print_matrix(locals()["matrix_2"], "Macierz 2")


if __name__ == "__main__":
    main()
