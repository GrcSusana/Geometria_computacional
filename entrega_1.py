# Susana Garcia Martin
# Geometria Computacional - Práctica 1. Código de Huffmann y Teorema de Shannon

# Voy a trabajar con la plantilla

import os
import numpy as np
import pandas as pd

#### Vamos al directorio de trabajo
os.getcwd()
#os.chdir(ruta)
#files = os.listdir(ruta)

with open('GCOM2024_pract1_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2024_pract1_auxiliar_esp.txt', 'r',encoding="utf8") as file:
      es = file.read()


#### Contamos cuantos caracteres hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

#### Transformamos en formato array de los carácteres (states) y su frecuencia
#### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))


"""
##### Para obtener una rama, fusionamos los dos states con menor frecuencia
distr = distr_en
''.join(distr['states'][[0,1]])

### Es decir:
states = np.array(distr['states'])
probab = np.array(distr['probab'])
state_new = np.array([''.join(states[[0,1]])])   #Ojo con: state_new.ndim
probab_new = np.array([np.sum(probab[[0,1]])])   #Ojo con: probab_new.ndim
codigo = np.array([{states[0]: 0, states[1]: 1}])
states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
distr = pd.DataFrame({'states': states, 'probab': probab, })
distr = distr.sort_values(by='probab', ascending=True)
distr.index=np.arange(0,len(states))

#Creamos un diccionario
branch = {'distr':distr, 'codigo':codigo}
"""


## Ahora definimos una función que haga exáctamente lo mismo
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab})
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
 
tree_en = huffman_tree(distr_en)
tree_en[0].items()
tree_en[0].values()

#Buscar cada estado dentro de cada uno de los dos items
list(tree_en[0].items())[0][1] ## Esto proporciona un '0'
list(tree_en[0].items())[1][1] ## Esto proporciona un '1'

# Igual para la clave en español

tree_es = huffman_tree(distr_es)
tree_es[0].items()
tree_es[0].values()

#Buscar cada estado dentro de cada uno de los dos items
list(tree_es[0].items())[0][1] ## Esto proporciona un '0'
list(tree_es[0].items())[1][1] 


# A partir de aqui ya no uso plantilla. Hay que recorrer el arbol que genera
# para que de un codigo para cada caracter

# idea: empezar a recorrer el arbol desde el final, viendo en qué lado del arbol
# va quedando e ir añadiendo 0 o 1. 
# primero hay que hacer una lista con todos los caracteres sin repeticiones
# luego recorrer esa lista a ir viendo en el árbol qué codigo se obtiene, 
# y añadir al diccionario
# Mejor recorremos el arbol desde el principio, pero hay que ir concatenando 
# por delante

# lo del codigo binario se puede coger la longitud truncada hacia arriba


def caracteres_sin_repeticion(texto):
    l = []
    for i in texto:
        if not i in l:
            l.append(i)
    return l


def crea_diccionario_clave(tree):
    lista_caracteres = caracteres_sin_repeticion(en)
    lista_arbol = list(tree)
    diccionario = {}
    for i in lista_arbol:
        [k0, k1] = list(i.keys())
        [v0, v1] = list(i.values())
        if len(k0) == 1:
            diccionario[k0] = str(v0)
        if len(k0) > 1:
            for j in k0:
                diccionario[j] = str(v0) + diccionario[j]
        if len(k1) == 1:
            diccionario[k1] = str(v1)
        if len(k1) > 1:
            for j in k1:
                diccionario[j] = str(v1) + diccionario[j]
    return diccionario

# Nos crea el diccionario para la clave en ingles
        
diccionario_en = crea_diccionario_clave(tree_en)

diccionario_es = crea_diccionario_clave(tree_es)


# Longitudes medias 

def longitud_media(distr, diccionario):
    lon = 0
    l_caracteres = list(distr['states'])
    l_prob = list(distr['probab']) 
    i = 0
    while i < len(l_caracteres):
        lon += len(diccionario[l_caracteres[i]])*l_prob[i]
        i += 1
    return lon

L_S_eng = longitud_media(distr_en, diccionario_en)

L_S_esp = longitud_media(distr_es, diccionario_es)


# Comprobar que se satisface el primer teorema de Shannon

def entropia(distr):
    ent = 0
    l_prob = list(distr['probab'])
    for i in l_prob:
        ent += i*np.log2(i)
    ent = ent * (-1)
    return ent


# Codificar y decodificar

def codificar(palabra, diccionario):
    r = ''
    for i in palabra:
        r = r + diccionario[i]
    return r

# Comprobar la eficiencia de longitud frente al codigo binario usual

def codificar_en_binario(palabra):
    # esta linea de codigo la he sacado de https://www.techiedelight.com/es/convert-string-to-binary-python/
    lista = list(format(letra, 'b') for letra in bytearray(palabra, "utf-8"))
    res = ''
    for i in lista:
        res = res + i
    return res
    

def decodificar(palabra_codificada, diccionario):
    r = ''
    i = 0 
    palabra = ''
    keys = list(diccionario.keys())
    vals = list(diccionario.values())
    while i < len(palabra_codificada):
        palabra = palabra + palabra_codificada[i]
        if palabra in vals:
            r = r + keys[vals.index(palabra)]
            palabra = ''
            i += 1
        else:
            i += 1
    return r
    
