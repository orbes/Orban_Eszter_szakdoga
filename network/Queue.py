class Queue:
    def __init__(self):
        self.input = []
        self.output = []

    def add(self, packet):
        self.input.append(packet)

    def copy_input_to_output(self):
        self.output.extend(self.input)
        self.input.clear()

    def size(self):
        return len(self.input) + len(self.output)

    def clear(self):
        self.input.clear()
        self.output.clear()

    def remove_first(self):
        if len(self.output) != 0:
            return self.output.pop(0)
        else:
            return -4

    def get_first(self):
        return self.output[0]

    def iterate_queue(self):
        for i in self.output:
            yield i
        for i in self.input:
            yield i
