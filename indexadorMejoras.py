import json
import os
import pickle
import re
from typing import Counter
from clize import run


from pr4 import diccionario
from pr5 import pesaje

from num2words import num2words


from nltk.stem import SnowballStemmer

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import spacy

stemmerES = SnowballStemmer("spanish")


#nltk.download('punkt')


with open(os.path.join("pr2","stop_es.pkl"), 'rb') as f:
    stop_es = pickle.load(f) 
def cargar_configuracion(ruta_archivo_config):
    with open(ruta_archivo_config) as f:
        config = json.load(f)
    return config    
def leer_documento_tsv(ruta):
    diccionario = {}
    with open(ruta, 'r', encoding='utf-8') as archivo:
        for linea in archivo:
            campos = linea.strip().split('\t')
            if len(campos) >= 2:
                clave = campos[0]
                valor = campos[1]
                diccionario[clave] = valor
    return diccionario
def convertir_numeros_a_letras(tokens):
    for i, token in enumerate(tokens):
        try:
            numero = float(token)
            if numero == float('inf') or numero == float('-inf'):
                tokens[i] = "infinito"
            elif abs(numero) > 10**24:  # Si el número es mayor que 10^24, lo consideramos demasiado grande
                tokens[i] = "Número demasiado grande"
            else:
                numero_entero = int(numero)
                if numero == numero_entero:
                    tokens[i] = num2words(numero_entero, lang='es')
                else:
                    tokens[i] = "No es un número entero"
        except ValueError:
            pass
    return tokens


def documentosMasRelevantes(similitud, id2doc, k=5):
    # Ordenar los documentos por similitud de mayor a menor
    documentos_ordenados = sorted(similitud.items(), key=lambda x: x[1], reverse=True)

    # Obtener los identificadores de los documentos más relevantes
    documentos_relevantes = [(id2doc[doc_id], sim) for doc_id, sim in documentos_ordenados[:k]]

    return documentos_relevantes

def procesar_documentos(documentos):
    for documento in documentos:
        # Convertir a minúsculas
        documentos[documento] = documentos[documento].lower()
        # Reemplazar comas por espacios
        documentos[documento] = documentos[documento].replace(',', ' ').replace('-',' ')
        # Eliminar caracteres no alfanuméricos y espacios
        documentos[documento] = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s]', '', documentos[documento])
        # Dividir en tokens
        documentos[documento] = documentos[documento].split()
        # Convertir números a letras
        documentos[documento] = convertir_numeros_a_letras(documentos[documento])
    return documentos

def procesar_nltk(documentos):
    for documento in documentos:
         # Convertir a minúsculas
        documentos[documento] = documentos[documento].lower()
        # Reemplazar comas por espacios
        documentos[documento] = documentos[documento].replace(',', ' ')
        # Eliminar caracteres no alfanuméricos y espacios
        documentos[documento] = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s]', '', documentos[documento])
        #tokenizar
        documentos[documento] =  word_tokenize(documentos[documento], language='spanish')
        # Convertir números a letras
        documentos[documento] = convertir_numeros_a_letras(documentos[documento])
    return documentos

def stopper(documentos):
    for documento in documentos:
        documentos[documento]  = [word for word in documentos[documento] if word not in stop_es and len(word)>1] 
    return documentos
def stemmer(documentos):
    stemmed_cache = {}  # Diccionario auxiliar para almacenar las palabras ya stemizadas
    for documento in documentos:
        documentos[documento] = [stemmed_cache[token] if token in stemmed_cache else stemmed_cache.setdefault(token, stemmerES.stem(str(token))) for token in documentos[documento]]
    return documentos
def stemmerRAE(documentos):
    nlp = spacy.load("es_core_news_sm")
    stemmed_cache = {}  # Diccionario auxiliar para almacenar las palabras ya stemizadas
    for documento in documentos:
        documentos[documento] = [stemmed_cache[token] 
                                 if token in stemmed_cache 
                                 else stemmed_cache.setdefault(token, nlp(token)) for token in documentos[documento]]
    return documentos

def kpalabrasComunes(documentos,k):
    kcomunes={}
    for documento in documentos:
        kcomunes[documento] = Counter(documentos[documento]).most_common(k)
        # Extraer solo los objetos sin la frecuencia
        kcomunes[documento] = [elemento for elemento, _ in kcomunes[documento]]
    return kcomunes

def kpalabrasPrimeras(documentos,k):
    kcomunes={}
    for documento in documentos:
        kcomunes[documento] = documentos[documento][:k]    
    return kcomunes  
 
def indexador(archivo_config: str):
    config = cargar_configuracion(archivo_config)
    documentos = leer_documento_tsv(config["rutaDocumentos"])
    contenido = documentos
    #procesar
    documentos = procesar_documentos(documentos)
    #stopper
    documentos = stopper(documentos)
    kcomunes = kpalabrasComunes(documentos,20)
    #stemmer 
    documentos = stemmer(documentos)

    #diccionarios
    term2id, id2term, doc2id, id2doc, inverted_index = diccionario.build_inverted_index(documentos) 
    #pesados
    docterms = pesaje.terms_in_documents(inverted_index)   
    estructura_normalizada = pesaje.divide_freq_by_max_term(docterms)
    matriz_pesos,idfs = pesaje.pesador(id2doc,estructura_normalizada,inverted_index)
    
    with open(config["rutaIndiceInvertido"], 'wb') as f:
        pickle.dump(inverted_index, f)
    with open(config["rutaTerm2id"], 'wb') as f:
        pickle.dump(term2id, f)  
    with open(config["rutaId2term"], 'wb') as f:
        pickle.dump(id2term, f)   
    with open(config["rutaDoc2id"], 'wb') as f:
        pickle.dump(doc2id, f)   
    with open(config["rutaId2doc"], 'wb') as f:
        pickle.dump(id2doc, f)   
    with open(config["rutaPesos"], 'wb') as f:
        pickle.dump(matriz_pesos,f) 
    with open(config["rutaIdfs"], 'wb') as f:
        pickle.dump(idfs,f) 
    with open(config["doc2term"], 'wb') as f:
        pickle.dump(docterms,f) 
    with open(config["rutakcomunes"], 'wb') as f:
        pickle.dump(kcomunes,f) 
    with open(config["rutaContenidoTexto"], 'wb') as f:
        pickle.dump(contenido,f)     

def runIndexador(archivo_config: str):
    """Indexa un csv de documentos"""
    #indexador(archivo_config)
    indexador(archivo_config)

if __name__ == "__main__":
    run(runIndexador)