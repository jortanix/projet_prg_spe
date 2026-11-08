import datetime
from collections import Counter

import numpy as np
from scipy import sparse

import pandas as pd

from Author import Author
from Document import Document, RedditDocument, ArxivDocument
from TextAnalyzer import TextAnalyzer

class Corpus:
    _instance = None

    def __new__(cls, nom="Mon corpus"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, nom="Mon corpus"):
        if hasattr(self, "_initialise"):
            return

        self.nom = nom
        self.documents = {}
        self.idDocument = 1
        self.authors = {}
        self.text_analyzer = TextAnalyzer()
        self.vocabulary = {}
        self.term_document_matrix = None
        self.search_engine = None
        self._initialise = True

    def reset(self, nom="Mon corpus"):
        """Réinitialise l'état du corpus (utile pour les tests et le rechargement)."""
        self.documents = {}
        self.idDocument = 1
        self.authors = {}
        self.vocabulary = {}
        self.term_document_matrix = None
        self.search_engine = None
        self.nom = nom

    def initialize_search_engine(self):
        """Initialise le moteur de recherche associé"""
        try:
            from SearchEngine import SearchEngine
            self.search_engine = SearchEngine(self)
        except ImportError:
            print("Attention: SearchEngine non disponible")
            self.search_engine = None

    def add(self, document):
        document_id = self.idDocument

        self.documents[document_id] = document

        auteur = document.auteur

        if auteur not in self.authors:
            self.authors[auteur] = Author(auteur)

        self.authors[auteur].add(document_id, document)

        self.idDocument += 1

        self._update_vocabulary(document.texte)
        self._build_term_document_matrix()

        if self.search_engine is None:
            self.initialize_search_engine()

    def show(self, n_docs=5):
        documents_tries = sorted(
            self.documents.values(),
            key=lambda document: document.date,
            reverse=True,
        )

        for document in documents_tries[:n_docs]:
            print(f"[{document.getType()}] {document}")

    def show_by_title(self, n_docs=5):
        documents_tries = sorted(
            self.documents.values(),
            key=lambda document: document.titre.lower(),
        )

        for document in documents_tries[:n_docs]:
            print(f"[{document.getType()}] {document}")

    def _update_vocabulary(self, text):
        if not self.vocabulary:
            all_texts = [doc.texte for doc in self.documents.values()]
            all_texts.append(text)
            self.vocabulary = self.text_analyzer.build_vocabulary(all_texts)
        else:
            new_tokens = self.text_analyzer.clean_text(text)
            for token in new_tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = 1
                else:
                    self.vocabulary[token] += 1

    def _build_term_document_matrix(self):
        if not self.documents:
            return

        texts = [doc.texte for doc in self.documents.values()]
        tfidf_scores = self.text_analyzer.compute_tf_idf(texts, self.vocabulary)

        num_docs = len(tfidf_scores)
        num_terms = len(self.vocabulary)

        rows = []
        cols = []
        data = []

        for i, doc_tfidf in enumerate(tfidf_scores):
            for j, term in enumerate(sorted(self.vocabulary.keys())):
                if term in doc_tfidf and doc_tfidf[term] > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(doc_tfidf[term])

        self.term_document_matrix = sparse.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_docs, num_terms)
        ).tocsr()

    def search(self, query: str, top_k: int = 5):
        """
        Recherche dans le corpus avec un mot-clé ou une phrase
        
        Args:
            query (str): Mot-clé ou phrase à rechercher
            top_k (int): Nombre de résultats à retourner
            
        Returns:
            List[Tuple[int, float]]: Liste des tuples (id_document, score)
        """
        if self.term_document_matrix is None or not self.vocabulary:
            return []

        query_tokens = self.text_analyzer.clean_text(query)

        # Vecteur de requête : 1 si le terme est présent, sinon 0
        query_vector = np.zeros(len(self.vocabulary))
        for i, term in enumerate(sorted(self.vocabulary.keys())):
            if term in query_tokens:
                query_vector[i] = 1.0

        query_sparse = sparse.csr_matrix(query_vector)

        # shape (nb_docs, 1), converti en vecteur 1D
        similarities = self.term_document_matrix.dot(query_sparse.T).toarray().ravel()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(list(self.documents.keys())[i], similarities[i])
                for i in top_indices if similarities[i] > 0]

    def get_document_similarity(self, doc_id1: int, doc_id2: int):
        """
        Calcule la similarité entre deux documents
        
        Args:
            doc_id1 (int): ID du premier document
            doc_id2 (int): ID du deuxième document
            
        Returns:
            float: Score de similarité entre 0 et 1
        """
        if self.term_document_matrix is None:
            return 0.0

        if doc_id1 not in self.documents or doc_id2 not in self.documents:
            return 0.0

        idx1 = list(self.documents.keys()).index(doc_id1)
        idx2 = list(self.documents.keys()).index(doc_id2)

        doc1_vector = self.term_document_matrix[idx1].toarray().ravel()
        doc2_vector = self.term_document_matrix[idx2].toarray().ravel()

        dot_product = float(np.dot(doc1_vector, doc2_vector))

        norm1 = np.linalg.norm(doc1_vector)
        norm2 = np.linalg.norm(doc2_vector)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_document_terms(self, doc_id: int, top_n: int = 10):
        """
        Récupère les termes les plus importants pour un document
        
        Args:
            doc_id (int): ID du document
            top_n (int): Nombre de termes à retourner
            
        Returns:
            List[Tuple[str, float]]: Liste des termes avec leurs scores
        """
        if self.term_document_matrix is None:
            return []

        if doc_id not in self.documents:
            return []

        idx = list(self.documents.keys()).index(doc_id)
        doc_vector = self.term_document_matrix[idx]

        terms_scores = []
        for i, term in enumerate(sorted(self.vocabulary.keys())):
            if doc_vector[0, i] > 0:
                terms_scores.append((term, doc_vector[0, i]))

        terms_scores.sort(key=lambda x: x[1], reverse=True)
        return terms_scores[:top_n]

    def __repr__(self):
        return (
            f"Corpus(nom='{self.nom}', "
            f"documents={len(self.documents)}, "
            f"auteurs={len(self.authors)})"
        )

    def concorde(self, expression, contexte=5):
        """Construit un concordancier : passages où apparaît l'expression,
        avec un contexte gauche/droit. Retourne un DataFrame pandas."""
        import re

        expr = re.compile(re.escape(expression), re.IGNORECASE)
        lignes = []

        for doc_id, doc in self.documents.items():
            texte = doc.texte
            for match in expr.finditer(texte):
                debut = max(0, match.start() - contexte)
                fin = min(len(texte), match.end() + contexte)
                lignes.append({
                    "document_id": doc_id,
                    "contexte_gauche": texte[debut:match.start()],
                    "motif": match.group(0),
                    "contexte_droit": texte[match.end():fin],
                })

        return pd.DataFrame(lignes)

    def stats(self, n=10):
        """Statistiques textuelles : nombre de mots différents et
        les n mots les plus fréquents du corpus."""
        compteur = Counter()

        for doc in self.documents.values():
            compteur.update(self.text_analyzer.clean_text(doc.texte))

        tableau = pd.DataFrame(
            list(compteur.items()),
            columns=["mot", "frequence"],
        )
        tableau = tableau.sort_values("frequence", ascending=False)

        differences = len(compteur)
        plus_frequents = tableau.head(n).to_dict("records")

        return {
            "nb_mots_differents": differences,
            "plus_frequents": plus_frequents,
        }

    def save(self, fichier):
        lignes = []

        for document_id, document in self.documents.items():
            ligne = {
                "id": document_id,
                "titre": document.titre,
                "auteur": document.auteur,
                "date": document.date.isoformat(),
                "url": document.url,
                "texte": document.texte,
                "type": document.getType(),
            }
            if document.getType() == "Reddit":
                ligne["nb_commentaires"] = document.get_nb_commentaires()
            elif document.getType() == "arXiv":
                ligne["co_auteurs"] = ",".join(document.get_co_auteurs())
            lignes.append(ligne)

        dataframe = pd.DataFrame(lignes)

        dataframe.to_csv(
            fichier,
            sep="\t",
            index=False,
            encoding="utf-8",
        )
        
        # Sauvegarder également le vocabulaire
        vocab_file = str(fichier).replace('.tsv', '_vocab.txt')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for term, freq in self.vocabulary.items():
                f.write(f"{term}\t{freq}\n")

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

            document_type = str(ligne.get("type", "")).strip()
            if document_type == "Reddit":
                document = RedditDocument(
                    titre=ligne["titre"],
                    auteur=ligne["auteur"],
                    date=datetime.datetime.fromisoformat(ligne["date"]),
                    url=ligne["url"],
                    texte=ligne["texte"],
                    nb_commentaires=int(ligne.get("nb_commentaires", 0) or 0),
                )
            elif document_type == "arXiv":
                co_auteurs = str(ligne.get("co_auteurs", "") or "").split(",")
                co_auteurs = [a for a in co_auteurs if a]
                document = ArxivDocument(
                    titre=ligne["titre"],
                    auteur=ligne["auteur"],
                    date=datetime.datetime.fromisoformat(ligne["date"]),
                    url=ligne["url"],
                    texte=ligne["texte"],
                    co_auteurs=co_auteurs,
                )
            else:
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

        # Reconstruire le vocabulaire et la matrice depuis les textes :
        # ils sont entièrement dérivables des documents et doivent refléter
        # le nettoyage courant du TextAnalyzer (stopwords, ponctuation...).
        all_texts = [doc.texte for doc in self.documents.values()]
        self.vocabulary = self.text_analyzer.build_vocabulary(all_texts)
        self._build_term_document_matrix()

        # Initialiser le moteur de recherche
        self.initialize_search_engine()