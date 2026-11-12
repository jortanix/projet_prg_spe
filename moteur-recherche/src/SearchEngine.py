import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from typing import List, Tuple, Dict, Any

class SearchEngine:
    """
    Classe principale du moteur de recherche avancé.
    Intègre toutes les fonctionnalités de recherche, analyse et visualisation.
    """
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.is_initialized = False
        self._initialize_engine()

    def _initialize_engine(self):
        if self.corpus and hasattr(self.corpus, 'term_document_matrix'):
            self.is_initialized = True
            print("Moteur de recherche initialisé avec succès")
        else:
            print("Erreur: Corpus non valide ou non initialisé")

    def advanced_search(self, query: str, filters: Dict = None,
                       sort_by: str = 'relevance', top_k: int = 10) -> List[Tuple[int, float, Dict]]:
        """Recherche avec filtres (auteur, type, période) et tri (pertinence, date, auteur)."""
        if not self.is_initialized:
            return []

        # On interroge plus de résultats que nécessaire pour pouvoir filtrer ensuite
        try:
            results = self.corpus.search(query, top_k=top_k * 2)
        except Exception as e:
            print(f"Erreur lors de la recherche: {e}")
            return []

        if filters:
            filtered_results = []
            for doc_id, score in results:
                doc = self.corpus.documents[doc_id]

                match = True
                if 'author' in filters and filters['author'] not in doc.auteur:
                    match = False
                if 'date_start' in filters:
                    # Defensive check for date existence and comparison
                    try:
                        if not hasattr(doc, 'date') or doc.date is None:
                            match = False
                        else:
                            if doc.date < filters['date_start']:
                                match = False
                    except (TypeError, AttributeError):
                        # Handle cases where date comparison fails or date is invalid
                        match = False
                if 'date_end' in filters:
                    # Defensive check for date existence and comparison
                    try:
                        if not hasattr(doc, 'date') or doc.date is None:
                            match = False
                        else:
                            if doc.date > filters['date_end']:
                                match = False
                    except (TypeError, AttributeError):
                        # Handle cases where date comparison fails or date is invalid
                        match = False
                if 'type' in filters and doc.getType() != filters['type']:
                    match = False

                if match:
                    filtered_results.append((doc_id, score))

            results = filtered_results

        if sort_by == 'date':
            # Defensive sorting with date validation
            def get_date_key(doc_id):
                doc = self.corpus.documents[doc_id]
                if hasattr(doc, 'date') and doc.date is not None:
                    return doc.date
                else:
                    # Return a very early date as fallback to put problematic docs at the beginning
                    import datetime
                    return datetime.datetime.min
            
            results.sort(key=lambda x: get_date_key(x[0]), reverse=True)
        elif sort_by == 'author':
            results.sort(key=lambda x: self.corpus.documents[x[0]].auteur)
        else:
            results.sort(key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in results[:top_k]:
            doc = self.corpus.documents[doc_id]
            # Defensive handling of date in metadata
            doc_date = getattr(doc, 'date', None)
            if doc_date is not None:
                try:
                    date_str = doc_date.isoformat() if hasattr(doc_date, 'isoformat') else str(doc_date)
                except:
                    date_str = str(doc_date)
            else:
                date_str = "Date non disponible"
            
            metadata = {
                'titre': doc.titre,
                'auteur': doc.auteur,
                'date': doc_date,
                'url': doc.url,
                'type': doc.getType(),
                'score': score
            }
            final_results.append((doc_id, score, metadata))

        return final_results

    def cluster_documents(self, n_clusters: int = 5) -> Dict[int, List[int]]:
        """Regroupe les documents similaires en clusters (K-means sur les vecteurs TF-IDF)."""
        if not self.is_initialized or self.corpus.term_document_matrix is None:
            return {}

        doc_vectors = self.corpus.term_document_matrix.toarray()

        try:
            centroids, _ = kmeans(doc_vectors, n_clusters)
            cluster_labels, _ = vq(doc_vectors, centroids)

            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(list(self.corpus.documents.keys())[i])

            return clusters
        except Exception as e:
            print(f"Erreur lors du clustering: {e}")
            return {}

    def get_document_recommendations(self, doc_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Recommende les documents les plus similaires à un document donné."""
        if not self.is_initialized:
            return []

        recommendations = []
        for other_doc_id in self.corpus.documents.keys():
            if other_doc_id != doc_id:
                similarity = self.corpus.get_document_similarity(doc_id, other_doc_id)
                if similarity > 0:
                    recommendations.append((other_doc_id, similarity))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def get_word_frequency_analysis(self, top_n: int = 20) -> Dict[str, int]:
        """Retourne les mots les plus fréquents de l'ensemble du corpus."""
        if not self.is_initialized:
            return {}

        word_freq = Counter()
        for doc in self.corpus.documents.values():
            word_freq.update(self.corpus.text_analyzer.clean_text(doc.texte))

        return dict(word_freq.most_common(top_n))

    def get_temporal_trends(self) -> Dict[str, List[Tuple[str, int]]]:
        """Regroupe les termes les plus importants par année de publication."""
        if not self.is_initialized:
            return {}

        trends = {}
        for doc_id, doc in self.corpus.documents.items():
            # Defensive handling of date
            if not hasattr(doc, 'date') or doc.date is None:
                continue
            try:
                year = doc.date.strftime("%Y")
            except:
                continue
            if year not in trends:
                trends[year] = Counter()

            important_terms = self.corpus.get_document_terms(doc_id, top_n=5)
            for term, _ in important_terms:
                trends[year][term] += 1

        return {year: terms.most_common(10) for year, terms in trends.items()}

    def evolution_temporelle(self, mot: str, score: str = 'tf_idf') -> List[Tuple[str, float]]:
        """Évolution de l'importance d'un mot au fil des années.

        Le corpus est découpé par année ; pour chaque année, on calcule
        l'importance moyenne du mot parmi les documents de cette année :
        fréquence brute ('freq') ou score TF-IDF moyen ('tf_idf').
        """
        if not self.is_initialized or not self.corpus.documents:
            return []

        tokens = self.corpus.text_analyzer.clean_text(mot)
        if not tokens:
            return []
        cible = tokens[0]

        # Regroupement des couples (doc_id, document) par année
        par_annee = {}
        for doc_id, doc in self.corpus.documents.items():
            # Defensive handling of date
            if not hasattr(doc, 'date') or doc.date is None:
                continue
            try:
                annee = doc.date.year
            except:
                continue
            par_annee.setdefault(annee, []).append((doc_id, doc))

        resultat = []
        for annee in sorted(par_annee):
            valeurs = []

            for doc_id, doc in par_annee[annee]:
                freq = self.corpus.text_analyzer.clean_text(doc.texte).count(cible)
                if freq == 0:
                    continue
                if score == 'freq':
                    valeurs.append(freq)
                else:
                    termes = self.corpus.get_document_terms(doc_id, top_n=10)
                    valeur_tf_idf = dict(termes).get(cible, 0.0)
                    valeurs.append(valeur_tf_idf)

            if valeurs:
                resultat.append(
                    (str(annee), round(float(sum(valeurs) / len(valeurs)), 4))
                )

        return resultat

    def export_results(self, results: List[Tuple[int, float, Dict]],
                      filename: str, format_type: str = 'csv'):
        """Exporte les résultats de recherche en CSV ou JSON."""
        import json
        import pandas as pd

        if format_type == 'csv':
            data = []
            for doc_id, score, metadata in results:
                data.append({
                    'document_id': doc_id,
                    'score': score,
                    'titre': metadata['titre'],
                    'auteur': metadata['auteur'],
                    'date': metadata['date'].isoformat(),
                    'url': metadata['url'],
                    'type': metadata['type']
                })

            pd.DataFrame(data).to_csv(filename, index=False)
            print(f"Résultats exportés vers {filename}")

        elif format_type == 'json':
            export_data = [
                {'document_id': doc_id, 'score': score, 'metadata': metadata}
                for doc_id, score, metadata in results
            ]

            def json_default(o):
                if hasattr(o, 'isoformat'):
                    return o.isoformat()
                return str(o)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=json_default)

            print(f"Résultats exportés vers {filename}")

    def _term_frequencies(self, doc_id):
        """Retourne les fréquences brutes des termes d'un document."""
        doc = self.corpus.documents[doc_id]
        tokens = self.corpus.text_analyzer.clean_text(doc.texte)
        return Counter(tokens)

    def search_bm25(self, query: str, top_k: int = 5,
                    k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
        """Recherche fondée sur la mesure Okapi BM25.

        k1 : saturation de la fréquence de terme, b : pondération de la longueur.
        """
        if not self.is_initialized or not self.corpus.documents:
            return []

        try:
            docs = list(self.corpus.documents.keys())
            n_docs = len(docs)
            longueurs = [len(self.corpus.text_analyzer.clean_text(
                self.corpus.documents[i].texte)) for i in docs]
            longueur_moyenne = sum(longueurs) / n_docs

            doc_term_freq = {i: self._term_frequencies(i) for i in docs}

            def document_frequency(terme):
                return sum(1 for tf in doc_term_freq.values() if tf.get(terme, 0) > 0)

            query_tokens = set(self.corpus.text_analyzer.clean_text(query))

            scores = []
            for i, doc_id in enumerate(docs):
                score = 0.0
                tf = doc_term_freq[doc_id]
                for terme in query_tokens:
                    f = tf.get(terme, 0)
                    if f == 0:
                        continue
                    df = document_frequency(terme)
                    idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                    denom = f + k1 * (1 - b + b * longueurs[i] / longueur_moyenne)
                    score += idf * (f * (k1 + 1)) / denom
                if score > 0:
                    scores.append((doc_id, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
        except Exception as e:
            print(f"Erreur lors de la recherche BM25: {e}")
            return []

    def analyser_sous_corpus(self, groupe_a: str, groupe_b: str,
                             top_n: int = 15) -> Dict[str, Any]:
        """Compare deux sous-corpus (groupes par type de document, ex. Reddit/arXiv).

        Chaque sous-corpus est traité comme un gros document et on compare
        l'importance des mots (TF-IDF) d'un côté et de l'autre.
        """
        if not self.is_initialized or not self.corpus.documents:
            return {}

        def texte_groupe(groupe):
            return " ".join(
                doc.texte for doc in self.corpus.documents.values()
                if doc.getType() == groupe
            )

        texte_a = texte_groupe(groupe_a)
        texte_b = texte_groupe(groupe_b)

        if not texte_a or not texte_b:
            return {}

        analyseur = self.corpus.text_analyzer
        vocab_ab = analyseur.build_vocabulary([texte_a, texte_b])
        scores_a = analyseur.compute_tf_idf([texte_a, texte_b], vocab_ab)[0]
        scores_b = analyseur.compute_tf_idf([texte_a, texte_b], vocab_ab)[1]

        def important(scores):
            return sorted(
                ((t, s) for t, s in scores.items() if s > 0),
                key=lambda x: x[1], reverse=True,
            )[:top_n]

        mots_a = {t for t, _ in important(scores_a)}
        mots_b = {t for t, _ in important(scores_b)}

        return {
            "mots_communs": sorted(mots_a & mots_b),
            "mots_specifiques_a": sorted(mots_a - mots_b),
            "mots_specifiques_b": sorted(mots_b - mots_a),
            "top_a": important(scores_a),
            "top_b": important(scores_b),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Produit des statistiques globales sur le corpus."""
        if not self.is_initialized:
            return {}

        stats = {
            'total_documents': len(self.corpus.documents),
            'total_authors': len(self.corpus.authors),
            'total_terms': len(self.corpus.vocabulary),
            'avg_document_length': 0,
            'most_active_author': '',
            'most_frequent_terms': [],
            'document_types': {}
        }

        if self.corpus.documents:
            lengths = [len(doc.texte) for doc in self.corpus.documents.values()]
            stats['avg_document_length'] = sum(lengths) / len(lengths)

        if self.corpus.authors:
            active_author = max(self.corpus.authors.values(),
                              key=lambda x: x.nbdocs)
            stats['most_active_author'] = active_author.name

        if self.corpus.vocabulary:
            sorted_terms = sorted(self.corpus.vocabulary.items(),
                                key=lambda x: x[1], reverse=True)
            stats['most_frequent_terms'] = sorted_terms[:10]

        doc_types = [doc.getType() for doc in self.corpus.documents.values()]
        stats['document_types'] = dict(Counter(doc_types))

        return stats