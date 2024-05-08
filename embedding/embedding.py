import csv
import pickle
import sys
import clize
from sentence_transformers import SentenceTransformer, util
import torch
sys.path.append('..\sistema_mejoras') 
import tools


def runEmbedding(archivo_config: str):
    """Convierte los documentos de un tsv en vectores con un sentence-transformer"""
    config = tools.cargar_configuracion(archivo_config)
    # Verificar si CUDA está disponible y seleccionar el dispositivo adecuado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cargar el modelo en CUDA
    modelo = "multi-qa-MiniLM-L6-cos-v1"
    embedder = SentenceTransformer(modelo, device=device)
    # Ruta al archivo TSV que contiene los documentos
    ruta_archivo = config["rutaDocumentos"]
    # Leer los documentos desde el archivo TSV
    corpus = tools.leer_documento_tsv(ruta_archivo) 
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    # Guardar el diccionario de embeddings en un archivo pickle
    with open(config["rutaEmbedding"], 'wb') as f:
        pickle.dump(corpus_embeddings, f)

if __name__ == '__main__':
    clize.run(runEmbedding)
