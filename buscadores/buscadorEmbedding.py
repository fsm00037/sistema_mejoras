
import pickle
import sys
import clize
import torch
from sentence_transformers import SentenceTransformer, util
sys.path.append('..\sistema_mejoras') 
import tools


def busquedaEmbedding(queries,top_k,archivo_config):
    config = tools.cargar_configuracion(archivo_config)
    # Verificar si CUDA está disponible y seleccionar el dispositivo adecuado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cargar el modelo en CUDA
    embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device=device)
    # Cargar el diccionario de embeddings de documentos desde el archivo pickle
    with open(config["rutaEmbedding"], 'rb') as f:
        embeddings_documentos = pickle.load(f)
    with open(config["rutaId2doc"], 'rb') as f:
        id2doc = pickle.load(f)
    # Resultados de las consultas
    resultados_consultas = {}
    for query in queries:
        query_embedding = embedder.encode(queries[query], convert_to_tensor=True)
        # Calcular la similitud del coseno entre la consulta y los embeddings de los documentos
        #cos_scores = util.cos_sim(query_embedding, embeddings_documentos)[0]
        hits = util.semantic_search(query_embedding, embeddings_documentos, top_k=top_k)
        # Seleccionar los resultados principales
        hits = hits[0]  # Tomar los resultados de la primera consulta
        hits = hits[:top_k]  # Limitar los resultados al top_k
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        documentos_similares = []
        for hit in hits:
            documentos_similares.append(id2doc[hit['corpus_id']])
        resultados_consultas[query] = documentos_similares
    
    return resultados_consultas
def runBusquedaEmbedding(archivo_config: str, top_k:int=10):
    """
    Sistema Embedding
    
    :param archivo_config: Archivo de configuración con las rutas.
    :param top_k: Número de documentos relevantes a recuperar.
    """
    config = tools.cargar_configuracion(archivo_config)
    rutaQueries = config["rutaQueris"]
    rutaResultado = config["rutaResultadoQueries"]
    queries = tools.leer_csv(rutaQueries,)
    resultado = busquedaEmbedding(queries,top_k,archivo_config)
    tools.crear_csv(resultado,rutaResultado)

if __name__ == '__main__':
    clize.run(runBusquedaEmbedding)

