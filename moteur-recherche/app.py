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
st.markdown(f"Corpus : **{corpus.nom}** -{len(corpus.documents)} documents, "
            f"{len({d.auteur for d in corpus.documents.values()})} auteurs.")

types_dispo = sorted({d.getType() for d in corpus.documents.values()})
auteurs_dispo = sorted({d.auteur for d in corpus.documents.values()})
dates = [d.date for d in corpus.documents.values()]
date_min, date_max = min(dates), max(dates)


def meta(doc_id):
    doc = corpus.documents[doc_id]
    return {
        "id": doc_id,
        "titre": doc.titre,
        "auteur": doc.auteur,
        "date": doc.date.strftime("%Y-%m-%d"),
        "type": doc.getType(),
    }


onglet_recherche, onglet_concor, onglet_temps, onglet_stats, onglet_comp = st.tabs(
    ["Recherche", "Concordancier", "Évolution temporelle", "Statistiques",
     "Analyse comparative"]
)

# ---------------------------------------------------------------------------
# Onglet Recherche (Bonus 1 : filtres source/auteur/période ; Bonus 2 : BM25)
# ---------------------------------------------------------------------------
with onglet_recherche:
    requete = st.text_input("Mots clés", key="req")
    nb_resultats = st.slider("Nombre d'articles à extraire", 1, 20, 5)

    with st.expander("Filtres de recherche", expanded=False):
        c1, c2, c3 = st.columns(3)
        type_filtre = c1.selectbox("Source", ["Toutes"] + types_dispo)
        auteur_filtre = c2.selectbox("Auteur", ["Tous"] + auteurs_dispo)
        tri = c3.selectbox("Tri", ["Pertinence", "Date", "Auteur"])
        c4, c5 = st.columns(2)
        d_debut = c4.date_input("Date début", date_min.date())
        d_fin = c5.date_input("Date fin", date_max.date())

    mode = st.radio("Mesure de score", ["TF-IDF (produit scalaire)", "Okapi BM25"],
                    horizontal=True)

    if st.button("Rechercher"):
        if not requete.strip():
            st.warning("Veuillez saisir des mots clés.")
        else:
            sort_map = {"Pertinence": "relevance", "Date": "date", "Auteur": "author"}
            if mode == "Okapi BM25":
                resultats = moteur.search_bm25(requete, top_k=nb_resultats)
                lignes = []
                for doc_id, score in resultats:
                    lignes.append({**meta(doc_id), "score_bm25": round(float(score), 4)})
            else:
                filtres = {}
                if type_filtre != "Toutes":
                    filtres["type"] = type_filtre
                if auteur_filtre != "Tous":
                    filtres["author"] = auteur_filtre
                if d_debut:
                    filtres["date_start"] = datetime.datetime.combine(d_debut, datetime.time.min)
                if d_fin:
                    filtres["date_end"] = datetime.datetime.combine(d_fin, datetime.time.max)
                resultats = moteur.advanced_search(
                    requete, filters=filtres,
                    sort_by=sort_map[tri], top_k=nb_resultats)
                lignes = []
                for doc_id, score, _ in resultats:
                    ligne = meta(doc_id)
                    ligne["score"] = round(float(score), 4)
                    lignes.append(ligne)

            if lignes:
                st.dataframe(pd.DataFrame(lignes), width='stretch')
            else:
                st.info("Aucun résultat ne correspond à la requête.")

# ---------------------------------------------------------------------------
# Onglet Concordancier (Bonus 8)
# ---------------------------------------------------------------------------
with onglet_concor:
    st.markdown("Affiche les passages où apparaît une expression, "
                "avec le contexte gauche / droit.")
    expression = st.text_input("Expression à chercher", key="expr")
    contexte = st.slider("Taille du contexte (caractères)", 1, 20, 6)

    if expression.strip():
        tableau = corpus.concorde(expression, contexte=contexte)
        if not tableau.empty:
            st.dataframe(tableau, width='stretch')
            st.caption(f"{len(tableau)} occurrence(s) dans le corpus.")
        else:
            st.info("L'expression n'apparaît dans aucun document.")
    else:
        st.write("Saisissez une expression pour construire le concordancier.")

# ---------------------------------------------------------------------------
# Onglet Évolution temporelle (Bonus 5)
# ---------------------------------------------------------------------------
with onglet_temps:
    st.markdown("Suit l'importance d'un mot au fil des années "
                "(fréquence brute ou score TF-IDF moyen par année).")
    mot = st.text_input("Mot à suivre", key="mot")
    type_score = st.radio("Mesure", ["tf_idf", "freq"], horizontal=True,
                          format_func=lambda x: "TF-IDF moyen" if x == "tf_idf" else "Fréquence")

    if mot.strip():
        serie = moteur.evolution_temporelle(mot.strip(), score=type_score)
        if serie:
            df = pd.DataFrame(serie, columns=["année", "score"]).set_index("année")
            st.line_chart(df)
            st.dataframe(df, width='stretch')
        else:
            st.info("Le mot n'apparaît dans aucun document du corpus.")
    else:
        st.write("Saisissez un mot pour afficher son évolution temporelle.")

# ---------------------------------------------------------------------------
# Onglet Statistiques (Bonus 7 : visualisation)
# ---------------------------------------------------------------------------
with onglet_stats:
    stats = moteur.get_statistics()
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", stats.get("total_documents", 0))
    col2.metric("Auteurs", stats.get("total_authors", 0))
    col3.metric("Termes du vocabulaire", stats.get("total_terms", 0))

    frequents = stats.get("most_frequent_terms", [])
    if frequents:
        df_freq = (pd.DataFrame(frequents[:15], columns=["mot", "frequence"])
                   .set_index("mot"))
        st.markdown("**Termes les plus fréquents**")
        st.bar_chart(df_freq)

# ---------------------------------------------------------------------------
# Onglet Analyse comparative (Bonus 4)
# ---------------------------------------------------------------------------
with onglet_comp:
    if len(types_dispo) >= 2:
        grp_a = st.selectbox("Sous-corpus A", types_dispo, key="a")
        grp_b = st.selectbox("Sous-corpus B",
                             [t for t in types_dispo if t != grp_a], key="b")
        comparaison = moteur.analyser_sous_corpus(grp_a, grp_b)
        if comparaison:
            st.markdown(f"**Mots communs :** {', '.join(comparaison['mots_communs'])}")
            st.markdown(f"**Spécifiques à {grp_a} :** "
                        f"{', '.join(comparaison['mots_specifiques_a'])}")
            st.markdown(f"**Spécifiques à {grp_b} :** "
                        f"{', '.join(comparaison['mots_specifiques_b'])}")
    else:
        st.info("Le corpus actuel ne contient qu'une seule source : "
                "l'analyse comparative est indisponible.")
