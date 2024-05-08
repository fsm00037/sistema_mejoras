import copy
import heapq
import math
import pickle
from clize import run
import sys

import clize
sys.path.append('../sistema_mejoras')  # Agrega la carpeta padre al path
import tools
#from mejoras import sinonimos 

import indexadorMejoras as im


def normalizarTokens(tokens,term2id,idfs):
    normas = {}
    norma = 0
    for token in tokens:
        if token in term2id:
            idf = idfs[term2id[token]] 
            normas[term2id[token]] =  idf
            norma += idf * idf
    norma = math.sqrt(norma)    
    for clave, valor in normas.items():
        normas[clave] =valor/norma
    return normas    

def calculoSimilitud(normasQuery,pesos):
    similitud = {}
    for idTerm, norma in normasQuery.items():
        if idTerm in pesos:
            for id_doc, peso in pesos[idTerm].items():
                if id_doc not in similitud: 
                    similitud[id_doc] = 0
                pesoDoc = norma * peso  
                similitud[id_doc] += pesoDoc  
    return similitud 
  
def busquedaTf_IDF(queriesAProcesar,k,archivo_config):
    config = tools.cargar_configuracion(archivo_config)
    with open(config["rutaPesos"], 'rb') as f:
        pesos = pickle.load(f)
    with open(config["rutaIdfs"], 'rb') as f:
        idfs = pickle.load(f)    
    with open(config["rutaTerm2id"], 'rb') as f:
        term2id = pickle.load(f)      
    with open(config["rutaId2doc"], 'rb') as f:
        id2doc = pickle.load(f)   
    queries = copy.deepcopy(queriesAProcesar)
    queries = im.procesar_documentos(queries)
    #for query in queries:
    #   queries[query] = queries[query] + sinonimos.expandir_con_sinonimos(queries[query])
    
    queries = im.stopper(queries)
    queries = im.stemmer(queries)
    
    normasQueries = {}
    for query in queries:
        normasQueries[query]  = normalizarTokens(queries[query],term2id,idfs)

    similitudes = {}
    topResultados = {}
    for norma in normasQueries:
        similitudes[norma] = calculoSimilitud(normasQueries[norma],pesos)
        similitudes[norma] = dict(heapq.nlargest(k, similitudes[norma].items(), key=lambda item: item[1]))
        topResultados[norma] = [id2doc[id] for id in similitudes[norma]]

    return topResultados

        

def runBusquedaTF(archivo_config: str,top_k:int = 10):
    """
    Sistema TF-IDF
    
    :param archivo_config: Archivo de configuración con las rutas.
    :param top_k: Número de documentos relevantes a recuperar.
    """
    config = tools.cargar_configuracion(archivo_config)
    rutaQueries = config["rutaQueris"]
    rutaResultado = config["rutaResultadoQueries"]
    queries = tools.leer_csv(rutaQueries)
    resultado = busquedaTf_IDF(queries,top_k,archivo_config)
    tools.crear_csv(resultado,rutaResultado)

if __name__ == '__main__':
    clize.run(runBusquedaTF)