import os



def replace_words_in_text(text, word_map):

    words = text.split()
    replaced_words = [word_map.get(word, word) for word in words]
    return " ".join(replaced_words)


input_file = "Tekst.txt"  # Plik wejściowy
output_file = "E_Tekst"  # Plik wyjściowy

word_map = {
    "jest": "to",
    "testowy": "przykładowy",
    "na": "dla"
}

if os.path.exists(input_file) and os.path.isfile(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    replaced_text = replace_words_in_text(text, word_map)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(replaced_text)

    print(f"Przetworzony tekst został zapisany w pliku: {output_file}")
else:
    print(f"Plik wejściowy '{input_file}' nie istnieje lub nie jest plikiem.")
