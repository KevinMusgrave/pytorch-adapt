class HookLogger:
    def __init__(self, name):
        self.name = name
        self.reset()

    def __call__(self, x):
        if self.str:
            self.str += "\n"
        self.str += f"{self.name}: {x}"

    def reset(self):
        self.str = ""
