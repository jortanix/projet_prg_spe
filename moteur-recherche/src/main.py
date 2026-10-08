import datetime
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

import praw
import xmltodict

from Author import Author
from Document import Document
from Corpus import Corpus


THEME = "artificial intelligence"
TAILLE_DOCS = 10


def nettoyer_texte(texte):
    return " ".join(texte.replace("\n", " ").split())


# ==================================================
# Initialisation des collections
# ==================================================

documents = {}
authors = {}
identifiant = 1


# ==================================================
# Acquisition Reddit
# ==================================================

reddit = praw.Reddit(
    client_id="m4W3R1MLvrnNI9LTbG35Eg",
    client_secret="ynb_xt5pKo4IFhssuuqr5ENTMu5Q4g",
    user_agent="Walid_M1",
)

subreddit = reddit.subreddit("artificial")
posts = list(subreddit.hot(limit=TAILLE_DOCS))

textes_reddit = []

for post in posts:
    titre = nettoyer_texte(post.title)
    contenu = nettoyer_texte(post.selftext)

    texte = titre

    if contenu:
        texte += ". " + contenu

    textes_reddit.append(texte)

    auteur = str(post.author) if post.author is not None else "[deleted]"

    date = datetime.datetime.fromtimestamp(
        post.created_utc,
        tz=datetime.timezone.utc,
    )

    url_post = f"https://www.reddit.com{post.permalink}"

    document = Document(
        titre=titre,
        auteur=auteur,
        date=date,
        url=url_post,
        texte=texte,
    )

    documents[identifiant] = document

    if auteur not in authors:
        authors[auteur] = Author(auteur)

    authors[auteur].add(identifiant, document)

    identifiant += 1

print(f"\nNombre de posts Reddit collectés : {len(textes_reddit)}")


# ==================================================
# Acquisition arXiv
# ==================================================

parametres = {
    "search_query": "all:artificial AND all:intelligence",
    "start": 0,
    "max_results": TAILLE_DOCS,
}

url = "https://arxiv.org/api/query?" + urllib.parse.urlencode(parametres)

print(f"\nURL arXiv utilisée :\n{url}")

textes_arxiv = []

try:
    time.sleep(3)

    requete_http = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/atom+xml, application/xml, text/xml, */*",
        },
    )

    with urllib.request.urlopen(requete_http, timeout=30) as response:
        data = response.read()

    data_dict = xmltodict.parse(data)
    entrees = data_dict["feed"].get("entry", [])

    if isinstance(entrees, dict):
        entrees = [entrees]

    for entree in entrees:
        titre = nettoyer_texte(entree["title"])
        resume = nettoyer_texte(entree["summary"])

        texte = f"{titre}. {resume}"
        textes_arxiv.append(texte)

        auteurs_arxiv = entree.get("author", [])

        if isinstance(auteurs_arxiv, dict):
            auteurs_arxiv = [auteurs_arxiv]

        noms_auteurs = []

        for auteur_arxiv in auteurs_arxiv:
            noms_auteurs.append(auteur_arxiv["name"])

        auteur = ", ".join(noms_auteurs)

        date = datetime.datetime.fromisoformat(
            entree["published"].replace("Z", "+00:00")
        )

        liens = entree.get("link", [])

        if isinstance(liens, dict):
            liens = [liens]

        url_article = entree["id"]

        for lien in liens:
            if lien.get("@rel") == "alternate":
                url_article = lien["@href"]
                break

        document = Document(
            titre=titre,
            auteur=auteur,
            date=date,
            url=url_article,
            texte=texte,
        )

        documents[identifiant] = document

        # Chaque article peut avoir plusieurs auteurs.
        for nom_auteur in noms_auteurs:
            if nom_auteur not in authors:
                authors[nom_auteur] = Author(nom_auteur)

            authors[nom_auteur].add(identifiant, document)

        identifiant += 1

    print(f"\nNombre d'articles arXiv collectés : {len(textes_arxiv)}")

except urllib.error.HTTPError as erreur:
    if erreur.code == 429:
        print("\nArXiv limite temporairement les requêtes : HTTP 429.")
    else:
        print(f"\nErreur HTTP lors de l'appel à arXiv : {erreur.code}")

except urllib.error.URLError as erreur:
    print(f"\nErreur réseau lors de l'appel à arXiv : {erreur.reason}")


# ==================================================
# Vérifications corpus et auteurs
# ==================================================

corpus = Corpus("Artificial Intelligence")

for document in documents.values():
    corpus.add(document)

print("\nCorpus créé :")
print(corpus)

print("\nCinq documents les plus récents :")
corpus.show(5)

dossier_data = Path(__file__).resolve().parent.parent / "data"
dossier_data.mkdir(exist_ok=True)

fichier_corpus = dossier_data / "corpus_objets.tsv"

corpus.save(fichier_corpus)
print(f"\nCorpus sauvegardé : {fichier_corpus}")

corpus_recharge = Corpus("Artificial Intelligence rechargé")
corpus_recharge.load(fichier_corpus)

print("Corpus rechargé :")
print(corpus_recharge)

print("\nCinq documents triés par titre :")
corpus.show_by_title(5)

docs = textes_reddit + textes_arxiv

print(f"\nTaille totale du corpus texte : {len(docs)} documents")
print(f"Nombre d'objets Document créés : {len(documents)}")
print(f"Nombre d'auteurs répertoriés : {len(authors)}")

print("\nCinq premiers auteurs :")

for auteur in list(authors.values())[:5]:
    print(auteur)

if authors:
    nom_premier_auteur = next(iter(authors))
    premier_auteur = authors[nom_premier_auteur]

    tailles_documents = []

    for document in premier_auteur.production.values():
        tailles_documents.append(len(document.texte))

    taille_moyenne = sum(tailles_documents) / len(tailles_documents)

    print(f"\nStatistiques pour : {premier_auteur.name}")
    print(f"Nombre de documents : {premier_auteur.nbdocs}")
    print(f"Taille moyenne : {taille_moyenne:.2f} caractères")