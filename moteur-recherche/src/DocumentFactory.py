from Document import RedditDocument, ArxivDocument


class DocumentFactory:
    """Crée l'instance adaptée selon la source du document."""

    @staticmethod
    def creer(source, **kwargs):
        if source == "reddit":
            nb_commentaires = kwargs.pop("nb_commentaires", 0)
            return RedditDocument(
                titre=kwargs["titre"],
                auteur=kwargs["auteur"],
                date=kwargs["date"],
                url=kwargs["url"],
                texte=kwargs["texte"],
                nb_commentaires=nb_commentaires,
            )

        if source == "arxiv":
            co_auteurs = kwargs.pop("co_auteurs", [])
            return ArxivDocument(
                titre=kwargs["titre"],
                auteur=kwargs["auteur"],
                date=kwargs["date"],
                url=kwargs["url"],
                texte=kwargs["texte"],
                co_auteurs=co_auteurs,
            )

        raise ValueError(f"Source inconnue : {source}")
