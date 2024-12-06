import json

class Person:
    def __init__(self, first_name, last_name, address, postal_code, pesel):
        self.first_name = first_name
        self.last_name = last_name
        self.address = address
        self.postal_code = postal_code
        self.pesel = pesel
    def to_json(self, file_path):
        data = {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "address": self.address,
            "postal_code": self.postal_code,
            "pesel": self.pesel
        }
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Dane zapisane do pliku {file_path}")
    @classmethod
    def from_json(cls, file_path):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        print(f"Dane odczytane z pliku {file_path}")
        return cls(
            data["first_name"],
            data["last_name"],
            data["address"],
            data["postal_code"],
            data["pesel"]
        )
    def __str__(self):

        return (f"Imię: {self.first_name}, Nazwisko: {self.last_name}, "
                f"Adres: {self.address}, Kod pocztowy: {self.postal_code}, PESEL: {self.pesel}")

if __name__ == "__main__":
    person = Person("Jan", "Kowalski", "ul. Lipowa 5", "00-001", "12345678901")
    print(person)
    person.to_json("person.json")
    new_person = Person.from_json("person.json")
    print(new_person)
