class State:

    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.transitions = {}

    def add_transition(self, input_symbol, next_state):

        self.transitions[input_symbol] = next_state

    def get_next_state(self, input_symbol):

        return self.transitions.get(input_symbol, None)

    def __repr__(self):
        return f"State({self.name}, Output: {self.output})"


class MooreMachine:

    def __init__(self):
        self.states = {}
        self.start_state = None

    def add_state(self, state, is_start=False):

        self.states[state.name] = state
        if is_start:
            self.start_state = state

    def run(self, inputs):

        if self.start_state is None:
            raise ValueError("Automat nie ma ustawionego stanu początkowego.")

        current_state = self.start_state
        outputs = [current_state.output]  

        for input_symbol in inputs:
            next_state = current_state.get_next_state(input_symbol)
            if next_state is None:
                raise ValueError(f"Brak przejścia dla stanu {current_state.name} i symbolu wejściowego '{input_symbol}'.")
            current_state = next_state
            outputs.append(current_state.output)

        return outputs


if __name__ == "__main__":
    s0 = State("S0", "A")
    s1 = State("S1", "B")
    s2 = State("S2", "C")

    s0.add_transition("0", s0)
    s0.add_transition("1", s1)
    s1.add_transition("0", s2)
    s1.add_transition("1", s0)
    s2.add_transition("0", s1)
    s2.add_transition("1", s2)

    moore_machine = MooreMachine()
    moore_machine.add_state(s0, is_start=True)
    moore_machine.add_state(s1)
    moore_machine.add_state(s2)

    input_sequence = "101010"
    output_sequence = moore_machine.run(input_sequence)

    print("Wejście:", input_sequence)
    print("Wyjście:", "".join(output_sequence))
