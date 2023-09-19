class Data:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '<' + self.name + '> ' + str(self.__dict__)
