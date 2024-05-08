
import clize
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import sys
sys.path.append('../sistema_mejoras')  # Agrega la carpeta padre al path
from buscadores import buscadorTF 
import tools
from embedding import crossEncoder


def buscaTF_RR(queries,top_k,topRR,archivo_config):
    
    # búsqueda tf-idf
    tf_retrieval = buscadorTF.busquedaTf_IDF(queries,topRR,archivo_config)

    resultados_consultas = crossEncoder.reRanking(tf_retrieval,queries,top_k,archivo_config)

    return resultados_consultas


def runbusquedaTF_RR(archivo_config: str,top_k: int = 10,rr:int = 32):
    """
    Sistema TF-IDF con Re-Ranking
    
    :param archivo_config: Archivo de configuración con las rutas.
    :param top_k: Número de documentos relevantes a recuperar.
    :param rr: Número de documentos a obtner en la primera recuperacion.
    """
    config = tools.cargar_configuracion(archivo_config)
    rutaQueries = config["rutaQueris"]
    rutaResultado = config["rutaResultadoQueries"]
    queries = tools.leer_csv(rutaQueries)
    resultado = buscaTF_RR(queries,top_k,rr,archivo_config)
    tools.crear_csv(resultado,rutaResultado)

if __name__ == '__main__':
    clize.run(runbusquedaTF_RR)