# Projet de Moteur de Recherche

Moteur de recherche permettant d'analyser, de structurer et d'interroger un corpus de documents textuels issus de Reddit et d'arXiv.

## Structure

```
moteur-recherche/
├── src/               # Code source
│   ├── main.py        # Script principal (collecte des données)
│   ├── Corpus.py      # Gestion du corpus et de la matrice TF-IDF
│   ├── Document.py    # Classes Document, RedditDocument, ArxivDocument
│   ├── Author.py      # Gestion des auteurs
│   ├── TextAnalyzer.py# Analyse textuelle (tokenisation, vocabulaire)
│   └── SearchEngine.py# Recherche avancée, clustering, recommandations
├── tests/             # Scripts de tests
└── data/              # Fichiers de sauvegarde (format TSV)
```

## Fonctionnalités

- Acquisition de documents depuis Reddit et arXiv
- Analyse textuelle : tokenisation, suppression des mots vides, construction du vocabulaire
- Matrice Documents × Termes avec scores TF-IDF calculés manuellement
- Moteur de recherche avec filtres (auteur, type, période) et tris (pertinence, date, auteur)
- Similarité entre documents
- Clustering (K-means), recommandations, analyse de fréquence, tendances temporelles, statistiques
- Export des résultats en CSV et JSON
- Sauvegarde et rechargement du corpus (format TSV)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Utilisation

```bash
cd moteur-recherche
python src/main.py
```

Exemple d'usage du moteur de recherche :

```python
from src.Corpus import Corpus

corpus = Corpus("Mon Corpus")
moteur = corpus.search_engine

# Recherche avec filtres et tri
moteur.advanced_search("intelligence artificielle", filters={'author': 'Jean Dupont'}, sort_by='date')

# Clustering
moteur.cluster_documents(n_clusters=5)

# Recommandations
moteur.get_document_recommendations(doc_id=1, top_k=5)
```

## Contraintes

Les bibliothèques scikit-learn et nltk ne sont pas utilisées. L'analyse textuelle et les scores TF-IDF sont implémentés manuellement ; NumPy, SciPy (matrices creuses, K-means) et Pandas sont utilisés comme support.

## Dépendances

Voir `requirements.txt`.
