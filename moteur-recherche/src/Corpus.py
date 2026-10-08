import datetime

import pandas as pd

from Author import Author
from Document import Document

class Corpus:
    def __init__(self, nom):
        self.nom = nom
        self.documents = {}
        self.idDocument = 1
        self.authors = {}

    def add(self, document):
        document_id = self.idDocument

        self.documents[document_id] = document

        auteur = document.auteur

        if auteur not in self.authors:
            self.authors[auteur] = Author(auteur)

        self.authors[auteur].add(document_id, document)

        self.idDocument += 1

    def show(self, n_docs=5):
        documents_tries = sorted(
            self.documents.values(),
            key=lambda document: document.date,
            reverse=True,
        )

        for document in documents_tries[:n_docs]:
            print(document)

    def show_by_title(self, n_docs=5):
        documents_tries = sorted(
            self.documents.values(),
            key=lambda document: document.titre.lower(),
        )

        for document in documents_tries[:n_docs]:
            print(document)

    def __repr__(self):
        return (
            f"Corpus(nom='{self.nom}', "
            f"documents={len(self.documents)}, "
            f"auteurs={len(self.authors)})"
        )

    def save(self, fichier):
        lignes = []

        for document_id, document in self.documents.items():
            lignes.append(
                {
                    "id": document_id,
                    "titre": document.titre,
                    "auteur": document.auteur,
                    "date": document.date.isoformat(),
                    "url": document.url,
                    "texte": document.texte,
                }
            )

        dataframe = pd.DataFrame(lignes)

        dataframe.to_csv(
            fichier,
            sep="\t",
            index=False,
            encoding="utf-8",
        )

    def load(self, fichier):
        dataframe = pd.read_csv(
            fichier,
            sep="\t",
            encoding="utf-8",
        )

        self.documents = {}
        self.authors = {}
        self.idDocument = 1

        for _, ligne in dataframe.iterrows():
            document_id = int(ligne["id"])

            document = Document(
                titre=ligne["titre"],
                auteur=ligne["auteur"],
                date=datetime.datetime.fromisoformat(ligne["date"]),
                url=ligne["url"],
                texte=ligne["texte"],
            )

            self.documents[document_id] = document

            if document.auteur not in self.authors:
                self.authors[document.auteur] = Author(document.auteur)

            self.authors[document.auteur].add(document_id, document)

            self.idDocument = max(self.idDocument, document_id + 1)