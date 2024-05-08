
from collections import Counter
import json
import math
import pickle
import time
from clize import run
import os


def imprimir_tamano_archivo(nombre_archivo):
    if os.path.isfile(nombre_archivo):
        tamano = os.path.getsize(nombre_archivo)
        print(f"El tamaño del archivo {nombre_archivo} es: {tamano} bytes")
    else:
        print(f"El archivo {nombre_archivo} no existe")
def cargar_configuracion(ruta_archivo_config):
    with open(ruta_archivo_config) as f:
        config = json.load(f)
    return config
#creamos una estructura para almacenar todos los terminos de cada documento junto su frecuencia
def terms_in_documents(inverted_index):     
    terms_by_document = {}

    # Recorrer el índice invertido
    for term_id, doc_freqs in inverted_index.items():
        for doc_id, freq in doc_freqs.items():
            if doc_id not in terms_by_document:
                terms_by_document[doc_id] = {}
            terms_by_document[doc_id][term_id] = freq

    return terms_by_document

def divide_freq_by_max_term(estructura_documentos):
    estructura_normalizada = {}

    # Iterar sobre la estructura de documentos
    for doc, terms in estructura_documentos.items():
        max_freq = max(terms.values())  # Encontrar la frecuencia máxima en el documento
        estructura_normalizada[doc] = {}
        # Dividir todas las frecuencias por la frecuencia máxima
        for term, freq in terms.items():
            estructura_normalizada[doc][term] = freq / max_freq

    return estructura_normalizada    

def pesador(id2doc,estructura_normalizada,inverted_index):
    #hacemos la matriz termino-documento
    n=len(id2doc)
    idfs = {}
    #matriz = np.zeros((len(id2term), n), dtype=float)
    diccionario = {}
    diccNormas = [0] * n
    for id_doc, terms in estructura_normalizada.items():
        for id_term, freq in terms.items():
            df = len(inverted_index[id_term])
            idf = math.log(n/df, 2)
            idfs[id_term] = idf
            peso = idf * freq
            #matriz[id_term][id_doc] = peso
            if id_term not in diccionario:
                diccionario[id_term] = {}
            diccionario[id_term][id_doc] = peso
            diccNormas[id_doc] += peso * peso
    # Calcular la norma de cada columna
    #normas_columnas = np.linalg.norm(matriz, axis=0)
    diccNormas = [math.sqrt(numero) for numero in diccNormas]
    # Dividir todos los elementos de la matriz por la norma de sus columnas
    #matrizNormalizada = matriz / normas_columnas
    for id_term, docs in diccionario.items():
        for id_doc, peso in docs.items():
            diccionario[id_term][id_doc] = peso/diccNormas[id_doc]
    return diccionario,idfs        
def run_pesos(archivo_config: str ,logs: bool = True):
    """Pesado de los documentos"""
    start_time = time.time()
    config = cargar_configuracion(archivo_config)
    with open(config["rutaIndiceInvertido"], 'rb') as f:
        inverted_index = pickle.load(f)
    with open(config["rutaId2term"], 'rb') as f:
        id2term = pickle.load(f)  
    with open(config["rutaId2doc"], 'rb') as f:
        id2doc = pickle.load(f)   
    docterms = terms_in_documents(inverted_index)   
    estructura_normalizada = divide_freq_by_max_term(docterms)

    #hacemos la matriz termino-documento
    n=len(id2doc)
    idfs = {}
    #matriz = np.zeros((len(id2term), n), dtype=float)
    diccionario = {}
    diccNormas = [0] * n
    for id_doc, terms in estructura_normalizada.items():
        for id_term, freq in terms.items():
            df = len(inverted_index[id_term])
            idf = math.log(n/df, 2)
            idfs[id_term] = idf
            peso = idf * freq
            #matriz[id_term][id_doc] = peso
            if id_term not in diccionario:
                diccionario[id_term] = {}
            diccionario[id_term][id_doc] = peso
            diccNormas[id_doc] += peso * peso
    # Calcular la norma de cada columna
    #normas_columnas = np.linalg.norm(matriz, axis=0)
    diccNormas = [math.sqrt(numero) for numero in diccNormas]
    # Dividir todos los elementos de la matriz por la norma de sus columnas
    #matrizNormalizada = matriz / normas_columnas
    for id_term, docs in diccionario.items():
        for id_doc, peso in docs.items():
            diccionario[id_term][id_doc] = peso/diccNormas[id_doc]
       
    with open(config["rutaPesos"], 'wb') as f:
        pickle.dump(diccionario,f) 
    with open(config["rutaIdfs"], 'wb') as f:
        pickle.dump(idfs,f) 
    end_time = time.time()
    indexing_time = end_time - start_time
    print(f"\tTiempo de pesado: {indexing_time} segundos")
    if logs:
        imprimir_tamano_archivo(config["rutaPesos"])
        imprimir_tamano_archivo(config["rutaIdfs"])
if __name__ == "__main__":
    run(run_pesos)        
