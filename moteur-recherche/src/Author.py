class Author:
    def __init__(self, name):
        self.name = name
        self.nbdocs = 0
        self.production = {}

    def add(self, document_id, document):
        self.production[document_id] = document
        self.nbdocs += 1

    def __str__(self):
        return f"{self.name} : {self.nbdocs} document(s)"