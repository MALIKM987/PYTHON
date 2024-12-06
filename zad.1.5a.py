import re
import os

def remove_words_with_re(text, words_to_remove):

    pattern = r'\b(' + '|'.join(re.escape(word) for word in words_to_remove) + r')\b'
    return re.sub(pattern, '', text).strip()


input_file = "Tekst.txt"
output_file = "E_Tekst"

words_to_remove = {"jest", "z", "na", "i"}

if os.path.exists(input_file) and os.path.isfile(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    filtered_text = remove_words_with_re(text, words_to_remove)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(filtered_text)

    print(f"Przefiltrowany tekst został zapisany w pliku: {output_file}")
else:
    print(f"Plik wejściowy '{input_file}' nie istnieje lub nie jest plikiem.")
