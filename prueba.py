#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script corregido para clustering difuso de artículos de becas.
Soluciona el problema de dimensiones en el algoritmo Fuzzy C-means.
"""

import os
import re
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo de texto de becas
archivo_txt = "corpus_txt/ayudas_21-22.txt"

# Carpeta de salida
carpeta_salida = "resultados_fuzzy_clusters"
os.makedirs(carpeta_salida, exist_ok=True)

# Importar el módulo de topic modeling
try:
    from transformer_topic_modeling import BecasTransformerTopicModel
    print("Módulo BecasTransformerTopicModel importado correctamente")
except ImportError as e:
    print(f"Error al importar BecasTransformerTopicModel: {e}")
    exit(1)

# Abrir y leer el archivo de texto
print(f"Leyendo archivo: {archivo_txt}")
try:
    with open(archivo_txt, 'r', encoding='utf-8') as f:
        texto_completo = f.read()
    print(f"Archivo leído correctamente: {len(texto_completo)} caracteres")
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    exit(1)

# Función para extraer artículos de manera robusta
def extraer_articulos(texto):
    articulos = []
    
    # Primer intento: buscar patrones "Artículo X."
    # Buscamos primero los inicios de todos los artículos
    inicios_articulos = []
    for match in re.finditer(r'Artículo\s+\d+\.', texto, re.IGNORECASE):
        inicios_articulos.append(match.start())
    
    print(f"Se encontraron {len(inicios_articulos)} inicios de artículos")
    
    if len(inicios_articulos) >= 5:  # Si encontramos suficientes artículos
        for i in range(len(inicios_articulos)):
            inicio = inicios_articulos[i]
            
            # El final del artículo es el inicio del siguiente artículo o el final del texto
            if i < len(inicios_articulos) - 1:
                fin = inicios_articulos[i+1]
            else:
                fin = len(texto)
            
            # Extraer el texto completo del artículo
            texto_articulo = texto[inicio:fin].strip()
            
            # Obtener el número y título del artículo
            match_info = re.match(r'Artículo\s+(\d+)\.', texto_articulo)
            if match_info:
                numero = match_info.group(1)
                
                # El título puede estar en la primera oración
                posible_titulo = texto_articulo.split('.')[0] + "."
                
                articulo = {
                    'numero': numero,
                    'titulo': posible_titulo,
                    'contenido': texto_articulo,
                    'texto_completo': texto_articulo
                }
                articulos.append(articulo)
    
    # Si no encontramos suficientes artículos, dividir por párrafos sustanciales
    if len(articulos) < 5:
        print("No se encontraron suficientes artículos, dividiendo por párrafos...")
        
        # Eliminar las líneas de "DIRECCIÓN DE VALIDACIÓN" que son repetitivas
        texto_limpio = re.sub(r'DIRECCIÓN DE VALIDACIÓN.*?\n', '', texto, flags=re.MULTILINE)
        
        # Dividir por bloques de texto separados por líneas en blanco
        bloques = [b.strip() for b in re.split(r'\n\s*\n', texto_limpio) if len(b.strip()) > 200]
        
        print(f"Se encontraron {len(bloques)} bloques de texto sustanciales")
        
        # Combinar bloques cercanos para formar "pseudo-artículos"
        pseudo_articulos = []
        pseudo_articulo_actual = ""
        
        for bloque in bloques:
            # Si el bloque parece ser un título de artículo o sección
            if re.match(r'(Artículo|CAPÍTULO|SECCIÓN)\s+', bloque, re.IGNORECASE):
                # Guardar el artículo anterior si existe
                if pseudo_articulo_actual:
                    pseudo_articulos.append(pseudo_articulo_actual)
                # Iniciar nuevo artículo
                pseudo_articulo_actual = bloque
            else:
                # Añadir al artículo actual
                if pseudo_articulo_actual:
                    pseudo_articulo_actual += "\n\n" + bloque
                else:
                    pseudo_articulo_actual = bloque
        
        # Añadir el último artículo
        if pseudo_articulo_actual:
            pseudo_articulos.append(pseudo_articulo_actual)
        
        print(f"Se formaron {len(pseudo_articulos)} pseudo-artículos")
        
        # Convertir los pseudo-artículos al formato de artículos
        for i, texto_articulo in enumerate(pseudo_articulos):
            # Intentar extraer número de artículo si existe
            match_info = re.search(r'Artículo\s+(\d+)\.', texto_articulo)
            if match_info:
                numero = match_info.group(1)
                titulo = texto_articulo.split('\n')[0]
            else:
                numero = str(i+1)
                # Usar la primera línea como título
                lineas = texto_articulo.split('\n')
                titulo = lineas[0] if lineas else f"Sección {i+1}"
            
            articulo = {
                'numero': numero,
                'titulo': titulo,
                'contenido': texto_articulo,
                'texto_completo': texto_articulo
            }
            articulos.append(articulo)
    
    # Filtrar artículos muy cortos o repetitivos
    articulos_filtrados = []
    textos_vistos = set()
    
    for articulo in articulos:
        # Simplificar el texto para detectar duplicados
        texto_simple = ' '.join(articulo['texto_completo'].split()[:30])
        
        # Verificar que no sea repetitivo y tenga contenido sustancial
        if (len(articulo['texto_completo']) > 200 and 
            texto_simple not in textos_vistos and 
            "DIRECCIÓN DE VALIDACIÓN" not in texto_simple):
            
            articulos_filtrados.append(articulo)
            textos_vistos.add(texto_simple)
    
    print(f"Después de filtrar, quedan {len(articulos_filtrados)} artículos")
    return articulos_filtrados

# Extraer artículos
print("Extrayendo artículos del documento...")
articulos = extraer_articulos(texto_completo)

if not articulos:
    print("No se pudieron extraer artículos. Saliendo...")
    exit(1)

# Guardar todos los artículos en un solo archivo para referencia
archivo_todos = os.path.join(carpeta_salida, "todos_los_articulos.txt")
with open(archivo_todos, 'w', encoding='utf-8') as f:
    f.write(f"Total de artículos extraídos: {len(articulos)}\n\n")
    
    for i, articulo in enumerate(articulos):
        f.write(f"\n\n{'='*50}\n")
        f.write(f"ARTÍCULO {i+1}: {articulo['titulo']}\n\n")
        f.write(articulo['texto_completo'])

print(f"Todos los artículos guardados en: {archivo_todos}")

# Inicializar el modelo de topic modeling
print("Inicializando modelo de topic modeling...")
modelo = BecasTransformerTopicModel()

# Definir categorías para 3 clusters
categorias = [
    'requisitos_academicos',
    'requisitos_economicos',
    'procedimiento_y_plazos'
]

modelo.topic_categories = {
    'requisitos_academicos': [
        'nota media', 'calificaciones', 'créditos', 'evaluación', 
        'curso completo', 'rendimiento', 'graduado', 'universidad', 
        'bachillerato', 'máster', 'matrícula', 'asignaturas',
        'expediente', 'titulación', 'superado', 'aprobar', 'grado',
        'enseñanzas', 'módulos', 'estudiantes', 'educación'
    ],
    'requisitos_economicos': [
        'renta', 'ingresos', 'patrimonio', 'umbral', 
        'cuantía', 'deducciones', 'recursos económicos', 
        'compensación', 'tributación', 'familia numerosa', 
        'miembros computables', 'patrimonio familiar', 'ayudas',
        'euros', 'importe', 'beca de matrícula', 'beneficiarios',
        'financiación', 'presupuestos', 'subvenciones', 'precios'
    ],
    'procedimiento_y_plazos': [
        'solicitud', 'convocatoria', 'plazo', 'fecha límite', 
        'documentación', 'inscripción', 'registro', 'presentación', 
        'tramitación', 'sede electrónica', 'certificado', 
        'procedimiento', 'notificación', 'resolución', 'recurso',
        'reintegro', 'presentar', 'información', 'verificación',
        'impreso', 'comunicación', 'formulario'
    ]
}

# Cargar el documento en el modelo
modelo.raw_texts = [texto_completo]
modelo.doc_names = [Path(archivo_txt).stem]

# Procesar el documento para obtener embeddings
print("Computando embeddings...")
modelo._compute_embeddings()

# Extraer los textos de los artículos para calcular sus embeddings
textos_articulos = [articulo['texto_completo'] for articulo in articulos]

# Calcular embeddings para cada artículo
print("Calculando embeddings para cada artículo...")
if modelo.model is not None:
    embeddings_articulos = modelo.model.encode(textos_articulos)
    print(f"Embeddings calculados usando el modelo transformer")
elif hasattr(modelo, 'alternate_embedding_fn') and modelo.alternate_embedding_fn is not None:
    embeddings_articulos = modelo.alternate_embedding_fn(textos_articulos)
    print(f"Embeddings calculados usando la función alternativa")
else:
    embeddings_articulos = modelo._compute_tfidf_embeddings(textos_articulos)
    print(f"Embeddings calculados usando TF-IDF")

# Calcular embeddings para las categorías definidas
embeddings_categorias = {}
print("Calculando embeddings para categorías predefinidas...")
for categoria, keywords in modelo.topic_categories.items():
    if modelo.model is not None:
        # Calcular embedding para cada palabra clave
        keyword_embeddings = modelo.model.encode(keywords)
        # Calcular el embedding promedio para la categoría
        embeddings_categorias[categoria] = np.mean(keyword_embeddings, axis=0)
    elif hasattr(modelo, 'alternate_embedding_fn') and modelo.alternate_embedding_fn is not None:
        # Usar función alternativa
        keyword_embeddings = modelo.alternate_embedding_fn(keywords)
        embeddings_categorias[categoria] = np.mean(keyword_embeddings, axis=0)
    else:
        # Usar enfoque basado en frecuencia de palabras
        embeddings_categorias[categoria] = modelo._compute_tfidf_embeddings([" ".join(keywords)])[0]

# En lugar de usar Fuzzy C-means, calcularemos la similitud directamente
print("Calculando similitud de cada artículo con cada categoría...")
n_articulos = len(articulos)
n_categorias = len(categorias)

# Matriz de similitud: [n_articulos, n_categorias]
similitud_matriz = np.zeros((n_articulos, n_categorias))

for i, embedding_articulo in enumerate(embeddings_articulos):
    for j, categoria in enumerate(categorias):
        # Calcular similitud coseno entre el artículo y la categoría
        similitud = cosine_similarity([embedding_articulo], [embeddings_categorias[categoria]])[0][0]
        similitud_matriz[i, j] = similitud

# Normalizar para que la suma por fila sea 1 (convertir a "membresía difusa")
suma_filas = similitud_matriz.sum(axis=1, keepdims=True)
pertenencia_articulos = similitud_matriz / suma_filas

print("Matriz de pertenencia calculada")

# Determinar el umbral de pertenencia para considerar que un artículo pertenece a un cluster
umbral_pertenencia = 0.25  # Un artículo pertenece a un cluster si su pertenencia es >= 25%

# Analizar cuántos artículos pertenecen a múltiples clusters
articulos_multicluster = 0
for i in range(len(articulos)):
    clusters_articulo = [j for j in range(n_categorias) if pertenencia_articulos[i, j] >= umbral_pertenencia]
    if len(clusters_articulo) > 1:
        articulos_multicluster += 1

print(f"Artículos que pertenecen a múltiples clusters: {articulos_multicluster} ({articulos_multicluster/len(articulos)*100:.1f}%)")

# Para cada cluster, guardar los artículos que tienen una pertenencia significativa
print("Guardando artículos por cluster...")

# También guardamos una versión que muestre todos los artículos con su grado de pertenencia a cada cluster
df_pertenencia = pd.DataFrame(pertenencia_articulos, 
                             columns=categorias,
                             index=[f"Artículo {a['numero']}: {a['titulo']}" for a in articulos])

# Guardar el DataFrame como HTML para mejor visualización
html_file = os.path.join(carpeta_salida, "pertenencia_articulos_clusters.html")
df_pertenencia.to_html(html_file)

# También guardamos como CSV
csv_file = os.path.join(carpeta_salida, "pertenencia_articulos_clusters.csv")
df_pertenencia.to_csv(csv_file)

print(f"Tabla de pertenencia guardada en: {html_file} y {csv_file}")

# Crear un heatmap para visualizar la pertenencia
plt.figure(figsize=(12, max(8, len(articulos) * 0.3)))
sns.heatmap(pertenencia_articulos, 
           yticklabels=[f"Art. {a['numero']}" for a in articulos],
           xticklabels=categorias,
           cmap="YlGnBu",
           vmin=0, vmax=1)
plt.title("Grado de pertenencia de cada artículo a cada cluster")
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "heatmap_pertenencia.png"))
print(f"Heatmap guardado en: {os.path.join(carpeta_salida, 'heatmap_pertenencia.png')}")

# Para cada categoria, guardar los artículos que tienen una pertenencia significativa
# ... (el resto del código anterior permanece igual hasta la sección de guardar los clusters)

# Para cada categoria, guardar los 6 artículos con mayor pertenencia
for cluster_id, categoria in enumerate(categorias):
    # Obtener los scores de pertenencia para este cluster
    scores_cluster = pertenencia_articulos[:, cluster_id]
    
    # Obtener los índices de los 6 artículos con mayor score (orden descendente)
    top6_indices = np.argsort(scores_cluster)[::-1][:6]
    
    articulos_cluster = [(articulos[i], scores_cluster[i]) 
                        for i in top6_indices]
    
    # Ordenar artículos por grado de pertenencia (descendente) por si acaso
    articulos_cluster.sort(key=lambda x: x[1], reverse=True)
    
    # Crear nombre de archivo
    categoria_formato = categoria.replace('_', '-')
    nombre_archivo = f"{Path(archivo_txt).stem}_{categoria_formato}_articulos_fuzzy.txt"
    ruta_archivo = os.path.join(carpeta_salida, nombre_archivo)
    
    # Guardar artículos
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write(f"Documento: {Path(archivo_txt).stem}\n")
        f.write(f"Categoría: {categoria}\n")
        f.write(f"Top 6 artículos más relevantes:\n\n")
        
        # Guardar cada artículo con un separador claro y su grado de pertenencia
        for idx, (articulo, pertenencia) in enumerate(articulos_cluster, 1):
            f.write(f"\n\n{'='*50}\n")
            f.write(f"Artículo #{idx} en relevancia\n")
            f.write(f"{articulo['titulo']}\n")
            f.write(f"Grado de pertenencia: {pertenencia*100:.1f}%\n\n")
            f.write(articulo['contenido'])
    
    print(f"Cluster '{categoria}' guardado en {nombre_archivo} con los 6 artículos más relevantes")


# Crear un archivo resumen de los artículos que están en múltiples categorías
articulos_multi = []
for i, articulo in enumerate(articulos):
    clusters_articulo = []
    for j, categoria in enumerate(categorias):
        if pertenencia_articulos[i, j] >= umbral_pertenencia:
            clusters_articulo.append((categoria, pertenencia_articulos[i, j]))
    
    if len(clusters_articulo) > 1:
        articulos_multi.append((articulo, clusters_articulo))

# Guardar el resumen de artículos en múltiples categorías
if articulos_multi:
    multi_file = os.path.join(carpeta_salida, "articulos_multicategoria.txt")
    with open(multi_file, 'w', encoding='utf-8') as f:
        f.write(f"Artículos que pertenecen a múltiples categorías (umbral: {umbral_pertenencia*100}%)\n\n")
        
        for articulo, clusters in articulos_multi:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"{articulo['titulo']}\n\n")
            
            # Mostrar los clusters a los que pertenece
            f.write("Pertenece a las siguientes categorías:\n")
            for categoria, pertenencia in clusters:
                f.write(f"- {categoria}: {pertenencia*100:.1f}%\n")
            
            f.write("\n")
            f.write(articulo['contenido'])
    
    print(f"Artículos en múltiples categorías guardados en: {multi_file}")

print(f"\nProceso completado. Resultados guardados en: {carpeta_salida}")