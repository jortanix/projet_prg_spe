#!/usr/bin/env python3
"""
Script de test pour la partie 2 du projet de moteur de recherche.
Ce script vérifie que toutes les nouvelles fonctionnalités ont été implémentées correctement.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from Corpus import Corpus
from Document import RedditDocument, ArxivDocument
import datetime

def test_text_analysis():
    """Test l'analyse textuelle"""
    print("=== Test de l'analyse textuelle ===")
    
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
    
    print(f"Corpus créé avec {len(corpus.documents)} documents")
    print(f"Vocabulaire construit avec {len(corpus.vocabulary)} termes")
    
    # Vérifier que le vocabulaire contient des termes
    if corpus.vocabulary:
        print("Les termes du vocabulaire :")
        for i, (term, freq) in enumerate(list(corpus.vocabulary.items())[:10]):
            print(f"  {term}: {freq}")
    else:
        print("Erreur: Le vocabulaire est vide")
        return False
    
    return True

def test_matrix_construction():
    """Test la construction de la matrice Documents × Termes"""
    print("\n=== Test de la construction de la matrice ===")
    
    corpus = Corpus("Test Matrix")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="Machine Learning",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/test1",
        texte="Le machine learning est une technique importante de l'intelligence artificielle",
        nb_commentaires=2
    )
    
    doc2 = ArxivDocument(
        titre="Deep Learning",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/test2",
        texte="Le deep learning repose sur les réseaux de neurones profonds",
        co_auteurs=["Test Co-Author"]
    )
    
    corpus.add(doc1)
    corpus.add(doc2)
    
    # Vérifier que la matrice a été construite
    if corpus.term_document_matrix is not None:
        print("Matrice Documents × Termes construite avec succès")
        print(f"Dimensions : {corpus.term_document_matrix.shape}")
        print(f"Type de matrice : {type(corpus.term_document_matrix)}")
        return True
    else:
        print("Erreur: La matrice n'a pas été construite")
        return False

def test_search_functionality():
    """Test la fonctionnalité de recherche"""
    print("\n=== Test de la fonctionnalité de recherche ===")
    
    corpus = Corpus("Test Search")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="IA et Réseaux de Neurones",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/search1",
        texte="L'intelligence artificielle utilise les réseaux de neurones pour l'apprentissage automatique",
        nb_commentaires=1
    )
    
    doc2 = ArxivDocument(
        titre="Algorithmes d'Optimisation",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/search2",
        texte="Les algorithmes d'optimisation sont essentiels pour entraîner les modèles IA",
        co_auteurs=["Test Co-Author"]
    )
    
    corpus.add(doc1)
    corpus.add(doc2)
    
    # Tester la recherche
    try:
        results = corpus.search("intelligence artificielle", top_k=3)
        print(f"Résultats de recherche pour 'intelligence artificielle': {len(results)} documents trouvés")
        for doc_id, score in results:
            print(f"  Document {doc_id}: score={score:.4f}")
        
        if len(results) > 0:
            return True
        else:
            print("Aucun document trouvé lors de la recherche")
            return False
            
    except Exception as e:
        print(f"Erreur lors de la recherche: {e}")
        return False

def test_similarity():
    """Test le calcul de similarité entre documents"""
    print("\n=== Test du calcul de similarité ===")
    
    corpus = Corpus("Test Similarity")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="Introduction à l'IA",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/sim1",
        texte="L'intelligence artificielle est un domaine de l'informatique",
        nb_commentaires=1
    )
    
    doc2 = ArxivDocument(
        titre="Techniques d'IA",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/sim2",
        texte="Les techniques d'intelligence artificielle comprennent l'apprentissage automatique",
        co_auteurs=["Test Co-Author"]
    )
    
    corpus.add(doc1)
    corpus.add(doc2)
    
    # Calculer la similarité
    try:
        similarity = corpus.get_document_similarity(1, 2)
        print(f"Similarité entre documents 1 et 2: {similarity:.4f}")
        if similarity >= 0:
            return True
        else:
            print("Erreur: Similarité négative")
            return False
    except Exception as e:
        print(f"Erreur lors du calcul de similarité: {e}")
        return False

def test_document_terms():
    """Test l'extraction des termes importants"""
    print("\n=== Test de l'extraction des termes importants ===")
    
    corpus = Corpus("Test Terms")
    
    # Documents de test
    doc1 = RedditDocument(
        titre="IA et Apprentissage Automatique",
        auteur="Test Author",
        date=datetime.datetime.now(),
        url="http://example.com/terms1",
        texte="L'intelligence artificielle utilise l'apprentissage automatique pour résoudre des problèmes complexes",
        nb_commentaires=1
    )
    
    corpus.add(doc1)
    
    # Extraire les termes importants
    try:
        important_terms = corpus.get_document_terms(1, top_n=5)
        print(f"Termes importants pour le document 1:")
        for term, score in important_terms:
            print(f"  {term}: {score:.4f}")
        
        if len(important_terms) > 0:
            return True
        else:
            print("Aucun terme important trouvé")
            return False
    except Exception as e:
        print(f"Erreur lors de l'extraction des termes: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("Lancement des tests de la partie 2 du projet...")
    
    tests = [
        test_text_analysis,
        test_matrix_construction,
        test_search_functionality,
        test_similarity,
        test_document_terms
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
        print("🎉 Tous les tests ont réussi!")
        return True
    else:
        print("❌ Certains tests ont échoué.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)