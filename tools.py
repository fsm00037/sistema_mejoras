import csv
import json


def cargar_configuracion(ruta_archivo_config):
    with open(ruta_archivo_config) as f:
        config = json.load(f)
    return config

# Función para leer documentos desde un archivo TSV y devolver un diccionario con el nombre del documento y su contenido
def leer_documento_tsv(ruta):
    documentos = []
    with open(ruta, 'r', encoding='utf-8') as archivo:
        for linea in archivo:
            campos = linea.strip().split('\t')
            if len(campos) >= 2:
                
                contenido_documento = campos[1]  # Suponiendo que el texto del documento está en el segundo campo
                documentos.append(contenido_documento)
    return documentos

#Funcion que guarda los resultados de las búsquedas en un csv
def crear_csv(diccionario, nombre_archivo):
    with open(nombre_archivo, 'w', newline='', encoding='utf-8') as archivo_csv:
        escritor = csv.writer(archivo_csv)
        escritor.writerow(['ID', 'rel_docs'])
        for clave, lista in diccionario.items():
            escritor.writerow([clave, ' '.join(map(str, lista))])


def leer_csv(nombre_archivo):
    diccionario = {}
    with open(nombre_archivo, 'r', encoding='utf-8', newline='') as archivo_csv:
        lector = csv.reader(archivo_csv)
        for indice, fila in enumerate(lector):
            diccionario[indice] = fila[1]  # Asigna la consulta a la clave del diccionario
    return diccionario        