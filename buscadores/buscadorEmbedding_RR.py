import csv
import pickle
import clize
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import sys
sys.path.append('../sistema_mejoras')  # Agrega la carpeta padre al path
import tools
from buscadores import buscadorEmbedding
from embedding import crossEncoder


def buscaEmbedding_RR(queries,top_k,topRR,archivo_config):
    #topRR de archivos relevantes obtenidos por query
    embedding_retrieval = buscadorEmbedding.busquedaEmbedding(queries,topRR,archivo_config)
    #re-ranking
    resultados_consultas = crossEncoder.reRanking(embedding_retrieval,queries,top_k,archivo_config)
    return resultados_consultas

def runBusquedaEmbedding(archivo_config: str,top_k:int=10,rr:int =32):
    """
    Sistema Embedding con Re-Ranking
    
    :param archivo_config: Archivo de configuración con las rutas.
    :param top_k: Número de documentos relevantes a recuperar.
    :param rr: Número de documentos a obtner en la primera recuperacion.
    """
    config = tools.cargar_configuracion(archivo_config)
    rutaQueries = config["rutaQueris"]
    rutaResultado = config["rutaResultadoQueries"]
    queries = tools.leer_csv(rutaQueries,)
    resultado = buscaEmbedding_RR(queries,top_k,rr,archivo_config)
    tools.crear_csv(resultado,rutaResultado)

if __name__ == '__main__':
    clize.run(runBusquedaEmbedding)