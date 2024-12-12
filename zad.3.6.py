class UpperCaseDecorator:

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        if isinstance(result, str):
            return result.upper()
        return result



@UpperCaseDecorator
def print_message():
    return "Witaj, świecie!"


print(print_message())
