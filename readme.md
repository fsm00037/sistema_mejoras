# Sistema de Recuperación de Información (SRI)

## Introducción:

En este repositorio, encontrarás principalmente un indexador de documentos `indexadorMejoras.py` y un conjunto de sistemas de recuperación de informacion en la carpeta `buscadores`:

También se incluyen carpetas auxiliares:

- `data`
- `indices`
- `resultados`

Y los archivos:

- `config.json`
- `eval.py`
- `tools.py`

## Diferentes buscadores:

- `buscadorTF.py`: Básico TF-IDF.
  
- `buscadorTF_PRF.py`: TF-IDF con pseudo-realimentación por relevancia.

- `buscadorEmbedding.py`: Sistema por semántica.

- `buscadorTF_RR.py`: TF-IDF recupera con Re-Ranking.

- `buscadorEmbedding_RR.py`: Sistema por semántica con Re-Ranking.

- `buscadorHibrido.py`: Sistema que recupera n documentos por TF-IDF y m por Embedding

## Antes de comenzar:
- Añade tus documentos en tsv a la ruta deseada:
- Genera los indices usando `indexadorMejoras.py` y `embedding/embedding.py`.
    ```
    python indexadorMejoras.py [archivo configuración] 
    python embedding/embedding.py [archivo configuración] 
    ```


## Instrucciones para ejecutar cualquier `buscador[tipo].py`:
Utiliza para obtener más información:
```
python buscador[tipo].py -h
```