import os


def list_files_recursive(directory):
    if not os.path.exists(directory):
        print("Katalog '{directory}' nie istnieje.")
        return

    if not os.path.isdir(directory):
        print("Ścieżka '{directory}' nie jest katalogiem.")
        return

    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            print(full_path)
        elif os.path.isdir(full_path):
            list_files_recursive(full_path)



directory = "C:\AGH\Z1\Z2"
print("Pliki w katalogu ", directory)
list_files_recursive(directory)
