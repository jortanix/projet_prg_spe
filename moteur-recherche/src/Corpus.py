import datetime
import numpy as np
from scipy import sparse

import pandas as pd

from Author import Author
from Document import Document
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
        self.search_engine = None  # Référence au moteur de recherche
        self._initialise = True
    
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
        
        # Mettre à jour le vocabulaire avec le nouveau document
        self._update_vocabulary(document.texte)
        
        # Initialiser le moteur de recherche si ce n'est pas déjà fait
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
        """Met à jour le vocabulaire avec un nouveau texte"""
        if not self.vocabulary:
            # Initialisation du vocabulaire avec tous les documents
            all_texts = [doc.texte for doc in self.documents.values()]
            all_texts.append(text)
            self.vocabulary = self.text_analyzer.build_vocabulary(all_texts)
        else:
            # Mise à jour du vocabulaire avec le nouveau texte
            new_tokens = self.text_analyzer.clean_text(text)
            for token in new_tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = 1
                else:
                    self.vocabulary[token] += 1

    def _build_term_document_matrix(self):
        """Construit la matrice Documents × Termes"""
        if not self.documents:
            return
            
        # Récupérer tous les textes des documents
        texts = [doc.texte for doc in self.documents.values()]
        
        # Calculer les scores TF-IDF pour tous les documents
        tfidf_scores = self.text_analyzer.compute_tf_idf(texts, self.vocabulary)
        
        # Créer la matrice sparse (Documents × Termes)
        num_docs = len(tfidf_scores)
        num_terms = len(self.vocabulary)
        
        # Initialiser les matrices pour la matrice creuse
        rows = []
        cols = []
        data = []
        
        # Remplir la matrice avec les scores TF-IDF
        for i, doc_tfidf in enumerate(tfidf_scores):
            for j, term in enumerate(sorted(self.vocabulary.keys())):
                if term in doc_tfidf and doc_tfidf[term] > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(doc_tfidf[term])
        
        # Créer la matrice creuse
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
        if not self.term_document_matrix or not self.vocabulary:
            return []
            
        # Nettoyer la requête
        query_tokens = self.text_analyzer.clean_text(query)
        
        # Calculer le vecteur de requête (TF-IDF)
        query_vector = np.zeros(len(self.vocabulary))
        for i, term in enumerate(sorted(self.vocabulary.keys())):
            if term in query_tokens:
                # Pour simplifier, on attribue un score de 1 au terme trouvé
                # Dans une implémentation plus avancée, on pourrait calculer le TF-IDF
                query_vector[i] = 1.0
        
        # Convertir en vecteur sparse
        query_sparse = sparse.csr_matrix(query_vector)
        
        # Calculer les scores de similarité (produit scalaire)
        similarities = self.term_document_matrix.dot(query_sparse.T).A1
        
        # Trouver les meilleurs résultats
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Retourner les résultats avec leurs scores
        results = [(list(self.documents.keys())[i], similarities[i]) 
                  for i in top_indices if similarities[i] > 0]
        
        return results

    def get_document_similarity(self, doc_id1: int, doc_id2: int):
        """
        Calcule la similarité entre deux documents
        
        Args:
            doc_id1 (int): ID du premier document
            doc_id2 (int): ID du deuxième document
            
        Returns:
            float: Score de similarité entre 0 et 1
        """
        if not self.term_document_matrix:
            return 0.0
            
        if doc_id1 not in self.documents or doc_id2 not in self.documents:
            return 0.0
            
        # Récupérer les indices des documents
        idx1 = list(self.documents.keys()).index(doc_id1)
        idx2 = list(self.documents.keys()).index(doc_id2)
        
        # Calculer la similarité cosinus
        doc1_vector = self.term_document_matrix[idx1]
        doc2_vector = self.term_document_matrix[idx2]
        
        # Produit scalaire
        dot_product = doc1_vector.dot(doc2_vector.T).A1[0]
        
        # Normes des vecteurs
        norm1 = np.linalg.norm(doc1_vector.A1)
        norm2 = np.linalg.norm(doc2_vector.A1)
        
        # Similarité cosinus
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def get_document_terms(self, doc_id: int, top_n: int = 10):
        """
        Récupère les termes les plus importants pour un document
        
        Args:
            doc_id (int): ID du document
            top_n (int): Nombre de termes à retourner
            
        Returns:
            List[Tuple[str, float]]: Liste des termes avec leurs scores
        """
        if not self.term_document_matrix:
            return []
            
        if doc_id not in self.documents:
            return []
            
        # Récupérer l'indice du document
        idx = list(self.documents.keys()).index(doc_id)
        
        # Récupérer le vecteur du document
        doc_vector = self.term_document_matrix[idx]
        
        # Trouver les termes les plus importants
        terms_scores = []
        for i, term in enumerate(sorted(self.vocabulary.keys())):
            if doc_vector[0, i] > 0:
                terms_scores.append((term, doc_vector[0, i]))
                
        # Trier par score décroissant
        terms_scores.sort(key=lambda x: x[1], reverse=True)
        
        return terms_scores[:top_n]

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
            
        # Charger le vocabulaire
        vocab_file = str(fichier).replace('.tsv', '_vocab.txt')
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocabulary = {}
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        term, freq = parts
                        self.vocabulary[term] = int(freq)
        except FileNotFoundError:
            # Si le fichier de vocabulaire n'existe pas, reconstruire le vocabulaire
            all_texts = [doc.texte for doc in self.documents.values()]
            self.vocabulary = self.text_analyzer.build_vocabulary(all_texts)
            
        # Initialiser le moteur de recherche
        self.initialize_search_engine()