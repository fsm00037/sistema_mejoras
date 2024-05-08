import pickle
import sys
import clize
sys.path.append('../sistema_mejoras')  # Agrega la carpeta padre al path
from buscadores import buscadorTF 
import tools

def busquedaPseudo(queriesAProcesar,k1,k2,archivo_config):
    config = tools.cargar_configuracion(archivo_config)
    with open(config["rutakcomunes"], 'rb') as f:
        kcomunes = pickle.load(f)       
    topResultados = buscadorTF.busquedaTf_IDF(queriesAProcesar,k1,archivo_config)
    #Pseudo Realimentacion-----------------------------    
    nuevosTerminosBusqueda = {}
    for query in topResultados:
        nuevosTerminosBusqueda[query] = []
        for doc in topResultados[query]:
            nuevosTerminosBusqueda[query].extend(kcomunes[doc])
    #union de las nuevas palabras a las queries
    nuevaQuery = {}
    for query in queriesAProcesar:
        nuevaQuery[query] = queriesAProcesar[query] +" "+ ' '.join(nuevosTerminosBusqueda[query])
        
        
    topResultadosNuevos = buscadorTF.busquedaTf_IDF(nuevaQuery,k2,archivo_config)
    return topResultadosNuevos
        

def runBusquedaPRF(archivo_config: str, top_k:int=10, prf:int = 5):
    """
    Sistema TF-IDF con PRF
    
    :param archivo_config: Archivo de configuración con las rutas.
    :param top_k: Número de documentos relevantes a recuperar.
    :param prf: Número de documentos a obtner en la primera recuperacion.
    """
    config = tools.cargar_configuracion(archivo_config)
    rutaQueries = config["rutaQueris"]
    rutaResultado = config["rutaResultadoQueries"]
    queries = tools.leer_csv(rutaQueries)
    resultado = busquedaPseudo(queries,prf,top_k,archivo_config)
    tools.crear_csv(resultado,rutaResultado)

if __name__ == '__main__':
    clize.run(runBusquedaPRF)