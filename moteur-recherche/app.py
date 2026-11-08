# -*- coding: utf-8 -*-
"""Interface web Streamlit du moteur de recherche.

Lancement : streamlit run app.py
"""

import datetime
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Le code des modules utilise des imports relatifs au dossier src
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from Corpus import Corpus  # noqa: E402
from SearchEngine import SearchEngine  # noqa: E402


@st.cache_resource
def charger_corpus():
    """Charge le corpus depuis le fichier sauvegardé (pas d'appel API)."""
    fichier = Path(__file__).resolve().parent / "data" / "corpus_objets.tsv"
    corpus = Corpus("Corpus AI")
    if fichier.exists():
        corpus.load(fichier)
    return corpus


corpus = charger_corpus()
moteur = SearchEngine(corpus)

st.title("Moteur de recherche")

requete = st.text_input("Mots clés")
nb_resultats = st.slider("Nombre d'articles à extraire", 1, 20, 5)

mode = st.radio("Mesure de score", ["TF-IDF (cosinus)", "Okapi BM25"])

types_dispo = sorted({d.getType() for d in corpus.documents.values()})
type_filtre = st.selectbox("Source", ["Toutes"] + types_dispo)

if st.button("Rechercher"):
    if not requete.strip():
        st.warning("Veuillez saisir des mots clés.")
    else:
        filtres = {}
        if type_filtre != "Toutes":
            filtres["type"] = type_filtre

        if mode == "Okapi BM25":
            resultats = moteur.search_bm25(requete, top_k=nb_resultats)
            lignes = [
                {
                    "document_id": doc_id,
                    "titre": corpus.documents[doc_id].titre,
                    "auteur": corpus.documents[doc_id].auteur,
                    "score_bm25": round(float(score), 4),
                }
                for doc_id, score in resultats
            ]
        else:
            resultats = moteur.advanced_search(requete, filters=filtres,
                                               top_k=nb_resultats)
            lignes = [
                {
                    "document_id": doc_id,
                    "titre": meta["titre"],
                    "auteur": meta["auteur"],
                    "type": meta["type"],
                    "score": round(float(score), 4),
                }
                for doc_id, score, meta in resultats
            ]

        if lignes:
            st.dataframe(pd.DataFrame(lignes))
        else:
            st.info("Aucun résultat.")

st.markdown("---")
st.subheader("Statistiques du corpus")

stats = moteur.get_statistics()
col1, col2, col3 = st.columns(3)
col1.metric("Documents", stats.get("total_documents", 0))
col2.metric("Auteurs", stats.get("total_authors", 0))
col3.metric("Termes du vocabulaire", stats.get("total_terms", 0))

frequents = stats.get("most_frequent_terms", [])
if frequents:
    st.bar_chart(
        pd.DataFrame(frequents[:10], columns=["mot", "frequence"])
        .set_index("mot")
    )

st.markdown("---")
st.subheader("Analyse comparative des sous-corpus")

if len(types_dispo) >= 2:
    grp_a = st.selectbox("Sous-corpus A", types_dispo, key="a")
    grp_b = st.selectbox("Sous-corpus B",
                         [t for t in types_dispo if t != grp_a], key="b")
    comparaison = moteur.analyser_sous_corpus(grp_a, grp_b)
    if comparaison:
        st.write(f"Mots communs : **{', '.join(comparaison['mots_communs'])}**")
        st.write(f"Spécifiques à {grp_a} : "
                 f"{', '.join(comparaison['mots_specifiques_a'])}")
        st.write(f"Spécifiques à {grp_b} : "
                 f"{', '.join(comparaison['mots_specifiques_b'])}")
else:
    st.info("Le corpus actuel ne contient qu'une seule source : "
            "l'analyse comparative est indisponible.")
