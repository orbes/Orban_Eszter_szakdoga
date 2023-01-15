import os


class Logs:
    def __init__(self, file_name, file_path='./logs'):
        real_name = os.path.join(file_path, file_name)
        self.f = open(real_name, "a")
        self.f.truncate(0)

    def log(self, line):
        self.f.write(line)

    def close(self):
        self.f.close()

