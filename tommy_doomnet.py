class DoomNet:
    def __init__(self, learning_rate=0.01):
        self.alpha = learning_rate

    def print_hyperpar(self):
        print self.alpha
