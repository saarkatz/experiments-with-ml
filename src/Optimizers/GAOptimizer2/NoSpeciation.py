class NoSpeciation:
    def __init__(self):
        pass
    def execute(self, model, population):
        population[:] = [[c for s in population for c in s]]
