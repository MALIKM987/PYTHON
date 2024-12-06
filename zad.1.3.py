import os


def remove_words_from_text(text, words_to_remove):

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in words_to_remove]
    return " ".join(filtered_words)



input_file = "Tekst.txt"
output_file = "E_Tekst"


words_to_remove = {"jest", "z", "na", "i"}


if os.path.exists(input_file) and os.path.isfile(input_file):

    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()


    filtered_text = remove_words_from_text(text, words_to_remove)


    with open(output_file, "w", encoding="utf-8") as file:
        file.write(filtered_text)

    print(f"Przefiltrowany tekst został zapisany w pliku: {output_file}")
else:
    print(f"Plik wejściowy '{input_file}' nie istnieje lub nie jest plikiem.")
