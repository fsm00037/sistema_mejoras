
import pickle
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import sys
sys.path.append('../sistema_mejoras')  # Agrega la carpeta padre al path
import tools

def reRanking(retrievals,queries,top_k,archivo_config):
    config = tools.cargar_configuracion(archivo_config)
    with open(config["rutaDoc2id"], 'rb') as f:
     doc2id = pickle.load(f)
    # Verificar si CUDA está disponible y seleccionar el dispositivo adecuado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cargar el modelo CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',device=device)
    docs = tools.leer_documento_tsv(config["rutaDocumentos"])
    # Resultados de las consultas
    resultados_consultas = {}
    for query in queries:
        documentos_similares = retrievals[query]
        if not documentos_similares :
            documentos_similares = ['4474591'] 
        ##### Re-Ranking con CrossEncoder #####
        # Ahora, evaluar todos los pasajes recuperados con el cross-encoder
        cross_inp = [[queries[query], docs[doc2id[documentoID]]] for documentoID in documentos_similares]
        cross_scores = cross_encoder.predict(cross_inp)
        hits = []
        # Ordenar los resultados por las puntuaciones del cross-encoder
        for idx in range(len(cross_scores)):
            hits.append({'corpus_id': documentos_similares[idx], 'cross-score': cross_scores[idx]})
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        documentos_similares = [hit['corpus_id']for hit in hits]
        
        resultados_consultas[query] = documentos_similares[:top_k]  

    return resultados_consultas