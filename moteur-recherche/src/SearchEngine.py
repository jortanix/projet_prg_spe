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
        """
        Initialise le moteur de recherche avec un corpus
        
        Args:
            corpus: Instance de la classe Corpus
        """
        self.corpus = corpus
        self.is_initialized = False
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialise le moteur de recherche"""
        if self.corpus and hasattr(self.corpus, 'term_document_matrix'):
            self.is_initialized = True
            print("Moteur de recherche initialisé avec succès")
        else:
            print("Erreur: Corpus non valide ou non initialisé")
    
    def advanced_search(self, query: str, filters: Dict = None, 
                       sort_by: str = 'relevance', top_k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        Recherche avancée avec filtres et options de tri
        
        Args:
            query (str): Requête de recherche
            filters (Dict): Filtres pour limiter les résultats
            sort_by (str): Critère de tri ('relevance', 'date', 'author')
            top_k (int): Nombre de résultats à retourner
            
        Returns:
            List[Tuple[int, float, Dict]]: Liste des résultats avec métadonnées
        """
        if not self.is_initialized:
            return []
        
        # Effectuer la recherche de base
        results = self.corpus.search(query, top_k=top_k * 2)  # Plus de résultats pour le filtrage
        
        # Appliquer les filtres si fournis
        if filters:
            filtered_results = []
            for doc_id, score in results:
                doc = self.corpus.documents[doc_id]
                
                # Appliquer les filtres
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
        
        # Trier les résultats
        if sort_by == 'date':
            results.sort(key=lambda x: self.corpus.documents[x[0]].date, reverse=True)
        elif sort_by == 'author':
            results.sort(key=lambda x: self.corpus.documents[x[0]].auteur)
        else:  # relevance
            results.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les meilleurs résultats avec métadonnées
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
        """
        Regroupe les documents similaires en clusters
        
        Args:
            n_clusters (int): Nombre de clusters à former
            
        Returns:
            Dict[int, List[int]]: Dictionnaire des clusters (clé=cluster_id, valeur=liste_doc_ids)
        """
        if not self.is_initialized or self.corpus.term_document_matrix is None:
            return {}
        
        # Utiliser les vecteurs de documents pour le clustering
        doc_vectors = self.corpus.term_document_matrix.toarray()
        
        # Appliquer le clustering K-means
        try:
            centroids, _ = kmeans(doc_vectors, n_clusters)
            cluster_labels, _ = vq(doc_vectors, centroids)
            
            # Regrouper les documents par cluster
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
        """
        Recommende des documents similaires à celui donné
        
        Args:
            doc_id (int): ID du document de référence
            top_k (int): Nombre de recommandations
            
        Returns:
            List[Tuple[int, float]]: Liste des recommandations (id, score)
        """
        if not self.is_initialized:
            return []
            
        recommendations = []
        
        # Calculer la similarité avec tous les autres documents
        for other_doc_id in self.corpus.documents.keys():
            if other_doc_id != doc_id:
                similarity = self.corpus.get_document_similarity(doc_id, other_doc_id)
                if similarity > 0:  # Seulement les documents similaires
                    recommendations.append((other_doc_id, similarity))
        
        # Trier par similarité décroissante
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_k]
    
    def get_word_frequency_analysis(self, top_n: int = 20) -> Dict[str, int]:
        """
        Analyse de la fréquence des mots dans l'ensemble du corpus
        
        Args:
            top_n (int): Nombre de mots les plus fréquents à retourner
            
        Returns:
            Dict[str, int]: Dictionnaire des mots et leur fréquence
        """
        if not self.is_initialized:
            return {}
        
        # Compter tous les termes du vocabulaire
        word_freq = Counter()
        
        # Parcourir tous les documents
        for doc in self.corpus.documents.values():
            tokens = self.corpus.text_analyzer.clean_text(doc.texte)
            word_freq.update(tokens)
        
        # Retourner les mots les plus fréquents
        return dict(word_freq.most_common(top_n))
    
    def get_temporal_trends(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Analyse des tendances temporelles des documents
        
        Returns:
            Dict[str, List[Tuple[str, int]]]: Tendances par année/mois
        """
        if not self.is_initialized:
            return {}
        
        # Regrouper par année
        trends = {}
        
        for doc in self.corpus.documents.values():
            year = doc.date.strftime("%Y")
            if year not in trends:
                trends[year] = Counter()
            
            # Compter les mots clés importants
            important_terms = self.corpus.get_document_terms(doc.id, top_n=5)
            for term, _ in important_terms:
                trends[year][term] += 1
        
        # Convertir en format approprié
        result = {}
        for year, terms in trends.items():
            result[year] = terms.most_common(10)  # Top 10 termes par année
        
        return result
    
    def export_results(self, results: List[Tuple[int, float, Dict]], 
                      filename: str, format_type: str = 'csv'):
        """
        Exporte les résultats de recherche dans un fichier
        
        Args:
            results (List): Résultats de recherche
            filename (str): Nom du fichier
            format_type (str): Format d'exportation ('csv', 'json')
        """
        import json
        import pandas as pd
        
        if format_type == 'csv':
            # Créer un DataFrame
            data = []
            for doc_id, score, metadata in results:
                row = {
                    'document_id': doc_id,
                    'score': score,
                    'titre': metadata['titre'],
                    'auteur': metadata['auteur'],
                    'date': metadata['date'].isoformat(),
                    'url': metadata['url'],
                    'type': metadata['type']
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"Résultats exportés vers {filename}")
            
        elif format_type == 'json':
            # Convertir en JSON
            export_data = []
            for doc_id, score, metadata in results:
                export_data.append({
                    'document_id': doc_id,
                    'score': score,
                    'metadata': metadata
                })
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"Résultats exportés vers {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Récupère des statistiques globales sur le corpus
        
        Returns:
            Dict[str, Any]: Statistiques du corpus
        """
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
        
        # Longueur moyenne des documents
        if self.corpus.documents:
            lengths = [len(doc.texte) for doc in self.corpus.documents.values()]
            stats['avg_document_length'] = sum(lengths) / len(lengths)
        
        # Auteur le plus actif
        if self.corpus.authors:
            active_author = max(self.corpus.authors.values(), 
                              key=lambda x: x.nbdocs)
            stats['most_active_author'] = active_author.name
        
        # Termes les plus fréquents
        if self.corpus.vocabulary:
            sorted_terms = sorted(self.corpus.vocabulary.items(), 
                                key=lambda x: x[1], reverse=True)
            stats['most_frequent_terms'] = sorted_terms[:10]
        
        # Types de documents
        doc_types = [doc.getType() for doc in self.corpus.documents.values()]
        stats['document_types'] = dict(Counter(doc_types))
        
        return stats