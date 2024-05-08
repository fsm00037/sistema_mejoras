from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
def expandir_con_sinonimos(consulta):
    sinonimos_expandidos = []
    for palabra in consulta:
        # Obtener los synsets (conjuntos de sinónimos) de la palabra
        synsets = wordnet.synsets(palabra, lang='spa')
        if synsets:
            # Extraer sinónimos de los synsets
            sinonimos = set()
            for synset in synsets:
                for lemma in synset.lemmas(lang='spa'):
                    sinonimos.add(lemma.name())
            sinonimos_expandidos.extend(list(sinonimos))     
    return sinonimos_expandidos


