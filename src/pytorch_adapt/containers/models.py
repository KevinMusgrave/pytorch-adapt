from .base_container import BaseContainer


class Models(BaseContainer):
    def train(self):
        for v in self.values():
            v.train()

    def eval(self):
        for v in self.values():
            v.eval()

    def zero_grad(self):
        for v in self.values():
            v.zero_grad()
