class ComplexNumber:

    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        raise TypeError("Dodawanie możliwe tylko z inną liczbą zespoloną.")

    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        raise TypeError("Odejmowanie możliwe tylko z inną liczbą zespoloną.")

    def __str__(self):
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        return f"{self.real} - {-self.imag}i"

z1 = ComplexNumber(3, 4)
z2 = ComplexNumber(1, -2)

print("Liczba zespolona z1:", z1)
print("Liczba zespolona z2:", z2)

z3 = z1 + z2
print("z1 + z2 =", z3)

z4 = z1 - z2
print("z1 - z2 =", z4)
