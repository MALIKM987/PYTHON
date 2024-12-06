import re
import os

def replace_words_with_re(text, word_map):

    pattern = r'\b(' + '|'.join(re.escape(word) for word in word_map.keys()) + r')\b'

    def replace_match(match):
        return word_map[match.group(0)]

    return re.sub(pattern, replace_match, text)


input_file = "Tekst.txt"
output_file = "E_Tekst"


word_map = {
    "jest": "to",
    "testowy": "przykładowy",
    "na": "dla"
}

if os.path.exists(input_file) and os.path.isfile(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    replaced_text = replace_words_with_re(text, word_map)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(replaced_text)

    print(f"Przetworzony tekst został zapisany w pliku: {output_file}")
else:
    print(f"Plik wejściowy '{input_file}' nie istnieje lub nie jest plikiem.")
