from nltk.corpus import wordnet
import nltk

try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

print(get_synonyms("django"))