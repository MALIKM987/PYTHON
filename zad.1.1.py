import os


def count_files(directory):
    if not os.path.exists(directory):
        print(f"Katalog '{directory}' nie istnieje.")
        return 0
    if not os.path.isdir(directory):
        print(f"Ścieżka '{directory}' nie jest katalogiem.")
        return 0

    # Liczenie plików w katalogu
    return sum(1 for entry in os.listdir(directory) if os.path.isfile(os.path.join(directory, entry)))


# Przykład użycia
directory = "\AGH\Z1\Z2"
file_count = count_files(directory)
print(f"Liczba plików w katalogu '{directory}': {file_count}")
