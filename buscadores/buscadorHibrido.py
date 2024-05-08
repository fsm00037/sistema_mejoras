import clize
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import sys
sys.path.append('../sistema_mejoras')  # Agrega la carpeta padre al path
from buscadores import buscadorTF 
from buscadores import buscadorEmbedding
from embedding import crossEncoder
import tools




def busquedaHibrido(queries,k_tf,k_embedding,top_k,archivo_config):
    tf_retrieval = buscadorTF.busquedaTf_IDF(queries,k_tf, "config.json")
    print("busqueda TF-idf completada")
    embedding_retrieval = buscadorEmbedding.busquedaEmbedding(queries,k_embedding,archivo_config)
    print("busqueda Semantica completada")
    unionQueries = {}
    for query in queries:
        unionQueries[query] = list(set(tf_retrieval[query] + embedding_retrieval[query]))
    
        
    resultados_consultas = crossEncoder.reRanking(unionQueries,queries,top_k,archivo_config)    
    return resultados_consultas

def runBusquedaHibrida(archivo_config: str,top_k:int=10,top_TF:int = 16,top_Em:int =16):
    """
    Sistema Híbrido
    
    :param archivo_config: Archivo de configuración con las rutas.
    :param top_k: Número de documentos relevantes a recuperar.
    :param top_TF: Número de documentos a obtener con TF-IDF.
    :param top_Em: Número de documentos a obtener con Embedding.
    """
    config = tools.cargar_configuracion(archivo_config)
    rutaQueries = config["rutaQueris"]
    rutaResultado = config["rutaResultadoQueries"]
    queries = tools.leer_csv(rutaQueries,)
    resultado = busquedaHibrido(queries,top_TF,top_Em,top_k,archivo_config)
    tools.crear_csv(resultado,rutaResultado)

if __name__ == '__main__':
    clize.run(runBusquedaHibrida)

