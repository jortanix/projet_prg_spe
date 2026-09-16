import time
import urllib.parse
import urllib.request
import urllib.error

import praw
import xmltodict


THEME = "artificial intelligence"
TAILLE_DOCS = 10


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
    titre = post.title.replace("\n", " ").strip()
    contenu = post.selftext.replace("\n", " ").strip()

    texte = titre

    if contenu:
        texte += ". " + contenu

    textes_reddit.append(texte)

print(f"\nNombre de posts Reddit collectés : {len(textes_reddit)}")

for texte in textes_reddit[:5]:
    print("\n---")
    print(texte)


# ==================================================
# Acquisition arXiv
# ==================================================

requete = urllib.parse.quote(
    "all:artificial AND all:intelligence",
    safe=":"
)

url = (
    "http://export.arxiv.org/api/query?"
    f"search_query={requete}"
    f"&start=0&max_results={TAILLE_DOCS}"
)

textes_arxiv = []

try:
    # Pause avant l'appel pour respecter la cadence arXiv.
    time.sleep(3)

    requete_http = urllib.request.Request(
        url,
        headers={
            "User-Agent": "M1 academic project - artificial intelligence corpus"
        }
    )

    with urllib.request.urlopen(requete_http, timeout=30) as response:
        data = response.read()

    data_dict = xmltodict.parse(data)

    entrees = data_dict["feed"].get("entry", [])

    # Une réponse avec un seul article est parfois un dictionnaire.
    if isinstance(entrees, dict):
        entrees = [entrees]

    for entree in entrees:
        titre = entree["title"].replace("\n", " ").strip()
        resume = entree["summary"].replace("\n", " ").strip()

        texte = f"{titre}. {resume}"
        textes_arxiv.append(texte)

    print(f"\nNombre d'articles arXiv collectés : {len(textes_arxiv)}")

    for texte in textes_arxiv[:5]:
        print("\n---")
        print(texte)

except urllib.error.HTTPError as erreur:
    if erreur.code == 429:
        print("\nArXiv limite temporairement les requêtes : HTTP 429.")
        print("Le programme continue avec les textes Reddit.")
    else:
        print(f"\nErreur HTTP lors de l'appel à arXiv : {erreur.code}")

except urllib.error.URLError as erreur:
    print(f"\nErreur réseau lors de l'appel à arXiv : {erreur.reason}")


# ==================================================
# Corpus texte provisoire
# ==================================================

docs = textes_reddit + textes_arxiv

print(f"\nTaille totale du corpus provisoire : {len(docs)} documents")