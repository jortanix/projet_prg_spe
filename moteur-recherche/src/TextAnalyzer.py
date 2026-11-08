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
        pass

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)

        return [token.strip() for token in text.split() if token.strip()]

    def clean_text(self, text: str, remove_stopwords: bool = True,
                   remove_numbers: bool = True) -> List[str]:
        """Nettoie le texte : minuscules, ponctuation, stopwords et nombres."""
        tokens = self.tokenize(text)

        cleaned_tokens = []
        for token in tokens:
            if remove_numbers and token.isdigit():
                continue
            if remove_stopwords and token in self.STOPWORDS_FR:
                continue
            if re.match(r'^[a-zA-ZÀ-ÿ]+$', token):
                cleaned_tokens.append(token)

        return cleaned_tokens

    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """Construit le vocabulaire global avec les fréquences des termes."""
        vocabulary = Counter()

        for doc in documents:
            vocabulary.update(self.clean_text(doc))

        return dict(vocabulary)

    def compute_term_frequency(self, document: str, vocabulary: Dict[str, int]) -> Dict[str, float]:
        """Calcule la fréquence normalisée des termes d'un document."""
        tokens = self.clean_text(document)
        tf = Counter(tokens)
        total_terms = len(tokens)

        return {
            term: tf.get(term, 0) / total_terms if total_terms > 0 else 0
            for term in vocabulary.keys()
        }

    def compute_tf_idf(self, documents: List[str], vocabulary: Dict[str, int]) -> List[Dict[str, float]]:
        """Calcule les scores TF-IDF pour l'ensemble des documents."""
        N = len(documents)

        df = Counter()
        for doc in documents:
            df.update(set(self.clean_text(doc)))

        tfidf_scores = []
        for doc in documents:
            tf = self.compute_term_frequency(doc, vocabulary)
            tfidf = {}

            for term, tf_value in tf.items():
                # IDF sans division par zéro si le terme n'apparaît dans aucun document
                idf = 0
                if df[term] > 0:
                    idf = 1 + 1 / (1 + df[term])

                tfidf[term] = tf_value * idf

            tfidf_scores.append(tfidf)

        return tfidf_scores