# Projet de Moteur de Recherche

Moteur de recherche permettant d'analyser, de structurer et d'interroger un corpus de documents textuels issus de Reddit et d'arXiv.

## Structure

```
moteur-recherche/
├── src/                 # Code source
│   ├── main.py          # Script principal (collecte des données)
│   ├── Corpus.py        # Gestion du corpus, matrice TF-IDF, concordancier
│   ├── Document.py      # Classes Document, RedditDocument, ArxivDocument
│   ├── DocumentFactory.py# Fabrique de documents (patron Factory)
│   ├── Author.py        # Gestion des auteurs
│   ├── TextAnalyzer.py  # Analyse textuelle (tokenisation, vocabulaire)
│   └── SearchEngine.py  # Recherche, BM25, clustering, recommandations
├── app.py               # Interface web Streamlit (bonus)
├── tests/               # Scripts de tests
└── data/                # Fichiers de sauvegarde (format TSV)
```

## Fonctionnalités

- Acquisition de documents depuis Reddit et arXiv
- Analyse textuelle : tokenisation, suppression des mots vides, construction du vocabulaire
- Matrice Documents × Termes avec scores TF-IDF calculés manuellement
- Moteur de recherche avec filtres (auteur, type, période) et tris (pertinence, date, auteur)
- Mesure Okapi BM25 en complément du TF-IDF
- Concordancier et statistiques textuelles
- Similarité entre documents
- Clustering (K-means), recommandations, analyse de fréquence, tendances temporelles, statistiques
- Analyse comparative de deux sous-corpus (Reddit vs arXiv)
- Export des résultats en CSV et JSON
- Sauvegarde et rechargement du corpus (format TSV)
- Interface web Streamlit

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Utilisation

Collecte et sauvegarde des données :

```bash
cd moteur-recherche
python src/main.py
```

Interface web (bonus) :

```bash
cd moteur-recherche
streamlit run app.py
```

## Exemples

```python
import sys
sys.path.append("moteur-recherche/src")

from Corpus import Corpus
from SearchEngine import SearchEngine

corpus = Corpus("Mon Corpus")
corpus.load("moteur-recherche/data/corpus_objets.tsv")
moteur = SearchEngine(corpus)

# Recherche avec filtres et tri
moteur.advanced_search("intelligence artificielle", filters={'type': 'Reddit'})

# Recherche avec Okapi BM25
moteur.search_bm25("réseaux de neurones", top_k=5)

# Analyse comparative de deux sous-corpus
moteur.analyser_sous_corpus("Reddit", "arXiv")

# Clustering et recommandations
moteur.cluster_documents(n_clusters=5)
moteur.get_document_recommendations(doc_id=1, top_k=5)
```

## Contraintes

Les bibliothèques scikit-learn et nltk ne sont pas utilisées. L'analyse textuelle et les scores TF-IDF sont implémentés manuellement ; NumPy, SciPy (matrices creuses, K-means) et Pandas sont utilisés comme support.

## Dépendances

Voir `requirements.txt`.
