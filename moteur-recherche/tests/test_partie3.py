#!/usr/bin/env python3
"""
Script de test pour la partie 3 du projet de moteur de recherche.
Ce script vérifie que toutes les nouvelles fonctionnalités avancées ont été implémentées correctement.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from Corpus import Corpus
from Document import RedditDocument, ArxivDocument
import datetime

def test_search_engine_initialization():
    """Test de l'initialisation du SearchEngine"""
    print("=== Test d'initialisation du SearchEngine ===")
    
    # Créer un corpus avec quelques documents de test
    corpus = Corpus("Test Corpus")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="Introduction à l'Intelligence Artificielle",
        auteur="Jean Dupont",
        date=datetime.datetime.now(),
        url="http://example.com/doc1",
        texte="L'intelligence artificielle est un domaine fascinant qui étudie l'automatisation des tâches cognitives.",
        nb_commentaires=5
    )
    
    doc2 = ArxivDocument(
        titre="Apprentissage Profond et Réseaux de Neurones",
        auteur="Marie Curie",
        date=datetime.datetime.now(),
        url="http://example.com/doc2",
        texte="Les réseaux de neurones profonds permettent d'atteindre des performances remarquables dans de nombreux domaines.",
        co_auteurs=["Albert Einstein"]
    )
    
    # Ajouter les documents
    corpus.add(doc1)
    corpus.add(doc2)
    
    # Vérifier que SearchEngine est bien initialisé
    if corpus.search_engine is not None:
        print("✓ SearchEngine initialisé avec succès")
        return True
    else:
        print("✗ Erreur: SearchEngine non initialisé")
        return False

def test_advanced_search():
    """Test de la recherche avancée"""
    print("\n=== Test de la recherche avancée ===")
    
    corpus = Corpus("Test Advanced Search")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="IA et Réseaux de Neurones",
        auteur="Test Author",
        date=datetime.datetime(2023, 1, 15),
        url="http://example.com/search1",
        texte="L'intelligence artificielle utilise les réseaux de neurones pour l'apprentissage automatique",
        nb_commentaires=2
    )
    
    doc2 = ArxivDocument(
        titre="Algorithmes d'Optimisation",
        auteur="Test Author",
        date=datetime.datetime(2023, 6, 20),
        url="http://example.com/search2",
        texte="Les algorithmes d'optimisation sont essentiels pour entraîner les modèles IA",
        co_auteurs=["Test Co-Author"]
    )
    
    corpus.add(doc1)
    corpus.add(doc2)
    
    try:
        # Test recherche basique
        results = corpus.search_engine.advanced_search("intelligence artificielle", top_k=5)
        print(f"✓ Recherche basique: {len(results)} résultats trouvés")
        
        # Test recherche avec filtres
        filters = {'author': 'Test Author'}
        filtered_results = corpus.search_engine.advanced_search(
            "intelligence", filters=filters, top_k=5
        )
        print(f"✓ Recherche avec filtres: {len(filtered_results)} résultats trouvés")
        
        # Test recherche avec tri
        sorted_results = corpus.search_engine.advanced_search(
            "intelligence", sort_by='date', top_k=5
        )
        print(f"✓ Recherche avec tri: {len(sorted_results)} résultats trouvés")
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test de recherche avancée: {e}")
        return False

def test_clustering():
    """Test du clustering de documents"""
    print("\n=== Test du clustering de documents ===")
    
    corpus = Corpus("Test Clustering")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="Machine Learning",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/clust1",
        texte="Le machine learning est une technique importante de l'intelligence artificielle",
        nb_commentaires=1
    )
    
    doc2 = ArxivDocument(
        titre="Deep Learning",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/clust2",
        texte="Le deep learning repose sur les réseaux de neurones profonds",
        co_auteurs=["Test Co-Author"]
    )
    
    doc3 = RedditDocument(
        titre="Réseaux de Neurones",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/clust3",
        texte="Les réseaux de neurones sont à la base de l'apprentissage automatique",
        nb_commentaires=3
    )
    
    corpus.add(doc1)
    corpus.add(doc2)
    corpus.add(doc3)
    
    try:
        # Effectuer le clustering
        clusters = corpus.search_engine.cluster_documents(n_clusters=2)
        print(f"✓ Clustering effectué: {len(clusters)} clusters créés")
        
        # Afficher les clusters
        for cluster_id, doc_ids in clusters.items():
            print(f"  Cluster {cluster_id}: {len(doc_ids)} documents")
            
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test de clustering: {e}")
        return False

def test_recommendations():
    """Test du système de recommandations"""
    print("\n=== Test du système de recommandations ===")
    
    corpus = Corpus("Test Recommendations")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="Introduction à l'IA",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/rec1",
        texte="L'intelligence artificielle est un domaine de l'informatique",
        nb_commentaires=1
    )
    
    doc2 = ArxivDocument(
        titre="Techniques d'IA",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/rec2",
        texte="Les techniques d'intelligence artificielle comprennent l'apprentissage automatique",
        co_auteurs=["Test Co-Author"]
    )
    
    doc3 = RedditDocument(
        titre="Applications de l'IA",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/rec3",
        texte="L'intelligence artificielle est utilisée dans de nombreux domaines",
        nb_commentaires=2
    )
    
    corpus.add(doc1)
    corpus.add(doc2)
    corpus.add(doc3)
    
    try:
        # Obtenir les recommandations pour le premier document
        recommendations = corpus.search_engine.get_document_recommendations(1, top_k=2)
        print(f"✓ Recommandations générées: {len(recommendations)} recommandations")
        
        for doc_id, score in recommendations:
            print(f"  Document {doc_id} (score: {score:.4f})")
            
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test des recommandations: {e}")
        return False

def test_analysis_functions():
    """Test des fonctions d'analyse avancée"""
    print("\n=== Test des fonctions d'analyse avancée ===")
    
    corpus = Corpus("Test Analysis")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="IA et Apprentissage Automatique",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/anal1",
        texte="L'intelligence artificielle utilise l'apprentissage automatique pour résoudre des problèmes complexes",
        nb_commentaires=1
    )
    
    doc2 = ArxivDocument(
        titre="Réseaux de Neurones Profonds",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/anal2",
        texte="Les réseaux de neurones profonds sont utilisés pour l'apprentissage automatique",
        co_auteurs=["Test Co-Author"]
    )
    
    corpus.add(doc1)
    corpus.add(doc2)
    
    try:
        # Test analyse fréquence des mots
        word_freq = corpus.search_engine.get_word_frequency_analysis(top_n=10)
        print(f"✓ Analyse de fréquence: {len(word_freq)} mots différents")
        
        # Test tendances temporelles
        temporal_trends = corpus.search_engine.get_temporal_trends()
        print(f"✓ Analyse temporelle: {len(temporal_trends)} périodes")
        
        # Test statistiques
        stats = corpus.search_engine.get_statistics()
        print(f"✓ Statistiques récupérées: {len(stats)} métriques")
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test des analyses: {e}")
        return False

def test_export_function():
    """Test de l'exportation des résultats"""
    print("\n=== Test de l'exportation ===")
    
    corpus = Corpus("Test Export")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="Document de Test",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/export1",
        texte="Ceci est un document de test pour l'exportation",
        nb_commentaires=1
    )
    
    corpus.add(doc1)
    
    try:
        # Effectuer une recherche pour avoir des résultats
        results = corpus.search_engine.advanced_search("test", top_k=5)
        
        # Exporter en CSV
        corpus.search_engine.export_results(results, "test_results.csv", "csv")
        print("✓ Exportation en CSV réussie")
        
        # Exporter en JSON
        corpus.search_engine.export_results(results, "test_results.json", "json")
        print("✓ Exportation en JSON réussie")
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test d'exportation: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("Lancement des tests de la partie 3 du projet...")
    
    tests = [
        test_search_engine_initialization,
        test_advanced_search,
        test_clustering,
        test_recommendations,
        test_analysis_functions,
        test_export_function
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                failed += 1
                print("✗ FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED with exception: {e}")
    
    print(f"\n=== Résultats des tests ===")
    print(f"Tests réussis: {passed}")
    print(f"Tests échoués: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 Tous les tests de la partie 3 ont réussi!")
        return True
    else:
        print("❌ Certains tests de la partie 3 ont échoué.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)