import json
from dataclasses import dataclass, asdict

@dataclass
class Person:
    first_name: str
    last_name: str
    address: str
    postal_code: str
    pesel: str

    def to_json(self, file_path: str):

        with open(file_path, "w") as json_file:
            json.dump(asdict(self), json_file, indent=4)
        print(f"Dane zapisane do pliku {file_path}")

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        print(f"Dane odczytane z pliku {file_path}")
        return cls(**data)

if __name__ == "__main__":

    person = Person("Jan", "Kowalski", "ul. Lipowa 5", "00-001", "12345678901")
    print(person)

    person.to_json("person.json")

    new_person = Person.from_json("person.json")
    print(new_person)
