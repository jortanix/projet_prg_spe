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
        results = self.corpus.search(query, top_k=top_k * 2)

        if filters:
            filtered_results = []
            for doc_id, score in results:
                doc = self.corpus.documents[doc_id]

                match = True
                if 'author' in filters and filters['author'] not in doc.auteur:
                    match = False
                if 'date_start' in filters and doc.date < filters['date_start']:
                    match = False
                if 'date_end' in filters and doc.date > filters['date_end']:
                    match = False
                if 'type' in filters and doc.getType() != filters['type']:
                    match = False

                if match:
                    filtered_results.append((doc_id, score))

            results = filtered_results

        if sort_by == 'date':
            results.sort(key=lambda x: self.corpus.documents[x[0]].date, reverse=True)
        elif sort_by == 'author':
            results.sort(key=lambda x: self.corpus.documents[x[0]].auteur)
        else:
            results.sort(key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in results[:top_k]:
            doc = self.corpus.documents[doc_id]
            metadata = {
                'titre': doc.titre,
                'auteur': doc.auteur,
                'date': doc.date,
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
            year = doc.date.strftime("%Y")
            if year not in trends:
                trends[year] = Counter()

            important_terms = self.corpus.get_document_terms(doc_id, top_n=5)
            for term, _ in important_terms:
                trends[year][term] += 1

        return {year: terms.most_common(10) for year, terms in trends.items()}

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