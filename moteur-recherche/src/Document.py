class Document:
    def __init__(self, titre, auteur, date, url, texte):
        self.titre = titre
        self.auteur = auteur
        self.date = date
        self.url = url
        self.texte = texte

    def afficher(self):
        print(f"Titre : {self.titre}")
        print(f"Auteur : {self.auteur}")
        print(f"Date : {self.date}")
        print(f"URL : {self.url}")
        print(f"Texte : {self.texte}")

    def __str__(self):
        return self.titre


class RedditDocument(Document):
    def __init__(
        self,
        titre,
        auteur,
        date,
        url,
        texte,
        nb_commentaires,
    ):
        super().__init__(titre, auteur, date, url, texte)
        self.nb_commentaires = nb_commentaires

    def get_nb_commentaires(self):
        return self.nb_commentaires

    def set_nb_commentaires(self, nb_commentaires):
        self.nb_commentaires = nb_commentaires

    def __str__(self):
        return f"[Reddit] {self.titre}"


class ArxivDocument(Document):
    def __init__(
        self,
        titre,
        auteur,
        date,
        url,
        texte,
        co_auteurs,
    ):
        super().__init__(titre, auteur, date, url, texte)
        self.co_auteurs = co_auteurs

    def get_co_auteurs(self):
        return self.co_auteurs

    def set_co_auteurs(self, co_auteurs):
        self.co_auteurs = co_auteurs

    def __str__(self):
        return f"[arXiv] {self.titre}"