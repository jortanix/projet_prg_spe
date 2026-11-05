import re
import string
from collections import Counter
from typing import List, Dict, Set

class TextAnalyzer:
    """
    Classe pour l'analyse textuelle des documents.
    Permet l'extraction de termes, le nettoyage et la construction du vocabulaire.
    """
    
    # Stopwords français (à compléter selon les besoins)
    STOPWORDS_FR = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'dans', 'sur', 'sous',
        'avec', 'sans', 'pour', 'contre', 'entre', 'par', 'en', 'au', 'aux', 'à',
        'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'que', 'qui', 'où', 'quand',
        'comment', 'pourquoi', 'combien', 'quel', 'quelle', 'quels', 'quelles',
        'ce', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes',
        'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs', 'je', 'tu', 'il',
        'elle', 'nous', 'vous', 'ils', 'elles', 'on', 'y', 'en', 'm', 't', 's', 'l',
        'j', 'd', 'c', 'q', 'étais', 'étais', 'était', 'étions', 'étiez', 'étaient',
        'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez',
        'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent'
    }
    
    def __init__(self):
        """Initialise l'analyseur textuel"""
        pass
    
    def tokenize(self, text: str) -> List[str]:
        """
        Divise le texte en tokens (mots).
        
        Args:
            text (str): Le texte à tokeniser
            
        Returns:
            List[str]: Liste des tokens
        """
        # Convertir en minuscules
        text = text.lower()
        
        # Remplacer la ponctuation par des espaces
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
        # Diviser en mots et supprimer les espaces vides
        tokens = [token.strip() for token in text.split() if token.strip()]
        
        return tokens
    
    def clean_text(self, text: str, remove_stopwords: bool = True, 
                   remove_numbers: bool = True) -> List[str]:
        """
        Nettoie le texte en supprimant les stopwords, nombres et ponctuation.
        
        Args:
            text (str): Le texte à nettoyer
            remove_stopwords (bool): Si vrai, retire les stopwords
            remove_numbers (bool): Si vrai, retire les nombres
            
        Returns:
            List[str]: Liste des mots nettoyés
        """
        # Tokenisation
        tokens = self.tokenize(text)
        
        # Filtrer les tokens
        cleaned_tokens = []
        for token in tokens:
            # Supprimer les nombres si demandé
            if remove_numbers and token.isdigit():
                continue
                
            # Supprimer les stopwords si demandé
            if remove_stopwords and token in self.STOPWORDS_FR:
                continue
                
            # Ne garder que les tokens alphabétiques ou alphanumériques
            if re.match(r'^[a-zA-ZÀ-ÿ]+$', token):
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """
        Construit le vocabulaire à partir d'une liste de documents.
        
        Args:
            documents (List[str]): Liste des textes des documents
            
        Returns:
            Dict[str, int]: Vocabulaire avec les termes et leur fréquence globale
        """
        vocabulary = Counter()
        
        for doc in documents:
            tokens = self.clean_text(doc)
            vocabulary.update(tokens)
            
        return dict(vocabulary)
    
    def compute_term_frequency(self, document: str, vocabulary: Dict[str, int]) -> Dict[str, float]:
        """
        Calcule la fréquence des termes dans un document.
        
        Args:
            document (str): Le texte du document
            vocabulary (Dict[str, int]): Le vocabulaire global
            
        Returns:
            Dict[str, float]: Fréquence des termes dans le document
        """
        tokens = self.clean_text(document)
        tf = Counter(tokens)
        total_terms = len(tokens)
        
        # Normalisation des fréquences
        tf_normalized = {}
        for term in vocabulary.keys():
            tf_normalized[term] = tf.get(term, 0) / total_terms if total_terms > 0 else 0
            
        return tf_normalized
    
    def compute_tf_idf(self, documents: List[str], vocabulary: Dict[str, int]) -> List[Dict[str, float]]:
        """
        Calcule les scores TF-IDF pour tous les documents.
        
        Args:
            documents (List[str]): Liste des textes des documents
            vocabulary (Dict[str, int]): Le vocabulaire global
            
        Returns:
            List[Dict[str, float]]: Scores TF-IDF pour chaque document
        """
        # Calculer le nombre total de documents
        N = len(documents)
        
        # Calculer le nombre de documents contenant chaque terme
        df = Counter()
        for doc in documents:
            tokens = set(self.clean_text(doc))
            df.update(tokens)
        
        # Calculer les scores TF-IDF
        tfidf_scores = []
        for doc in documents:
            tf = self.compute_term_frequency(doc, vocabulary)
            tfidf = {}
            
            for term, tf_value in tf.items():
                # IDF = log(N / df(term))
                idf = 0
                if df[term] > 0:
                    idf = 1 + 1 / (1 + df[term])  # Formule alternative pour éviter division par zéro
                    
                tfidf[term] = tf_value * idf
                
            tfidf_scores.append(tfidf)
            
        return tfidf_scores