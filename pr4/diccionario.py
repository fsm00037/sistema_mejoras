import os
import json
from collections import defaultdict
import time
import pickle

from clize import run

def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        tokens = json.load(file)
        return os.path.basename(file_path), tokens

def build_inverted_index(docs):
    term2id = {}
    id2term = []
    doc2id = {}
    id2doc = []
    inverted_index = defaultdict(dict)
    term_id_counter = 0
    doc_id_counter = 0

    # Recorrer todos los archivos en la carpeta
    for doc in docs:
           
            doc_id = doc
            tokens = docs[doc]
            

            # Asignar un identificador único para el documento si no existe
            if doc_id not in doc2id:
                doc_id = doc_id_counter
                doc2id[doc] = doc_id
                id2doc.append(doc)
                doc_id_counter += 1

            for token in tokens:
                # Asignar un identificador único para el término si no existe
                if token not in term2id:
                    term2id[token] = term_id_counter
                    id2term.append(token)
                    term_id_counter += 1

                term_id = term2id[token]
                doc_id = doc2id[doc]

                # Incrementar la frecuencia del término en el documento
                if doc_id in inverted_index[term_id]:
                    inverted_index[term_id][doc_id] += 1
                else:
                    inverted_index[term_id][doc_id] = 1

    return term2id, id2term, doc2id, id2doc, inverted_index
def imprimir_tamano_archivo(nombre_archivo):
    if os.path.isfile(nombre_archivo):
        tamano = os.path.getsize(nombre_archivo)
        print(f"El tamaño del archivo {nombre_archivo} es: {tamano} bytes")
    else:
        print(f"El archivo {nombre_archivo} no existe")
def runInvertexIndex(archivo_config: str ,logs: bool = True):
    """Crea el indice invertido"""
    start_time = time.time()
    config = cargar_configuracion(archivo_config)
    folder_path = config["rutaTokensParaIndexar"]  # Ruta de la carpeta que contiene los archivos JSON
    term2id, id2term, doc2id, id2doc, inverted_index = build_inverted_index(folder_path)
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
    end_time = time.time()
    indexing_time = end_time - start_time
    print(f"\tTiempo de indexación: {indexing_time} segundos")
    if logs:
        imprimir_tamano_archivo(config["rutaIndiceInvertido"]);
        imprimir_tamano_archivo(config["rutaTerm2id"]);
        imprimir_tamano_archivo(config["rutaId2term"]);
        imprimir_tamano_archivo(config["rutaDoc2id"]);
        imprimir_tamano_archivo(config["rutaId2doc"]);
    

def cargar_configuracion(ruta_archivo_config):
    with open(ruta_archivo_config) as f:
        config = json.load(f)
    return config
# Ejemplo de uso

if __name__ == "__main__":
   run(runInvertexIndex)
    

        
            
    
    