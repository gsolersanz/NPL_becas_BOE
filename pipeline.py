#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline completo para análisis de documentos de becas:
1. Extracción de artículos
2. Topic modeling y clustering
3. Generación de resúmenes con diferentes modelos
4. Evaluación y comparación de resúmenes usando LLM

Uso:
    python pipeline.py --input ruta/al/archivo.txt --output carpeta_resultados 
                       --summarization_models bart t5 --evaluate
"""

import os
import sys
import re
import argparse
import shutil
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Importar módulos necesarios
try:
    from transformer_topic_modeling import BecasTransformerTopicModel
    from summarization_models import ClusterSummarizer
    from llm_evaluator import compare_summaries
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que transformer_topic_modeling.py, summarization_models.py y llm_evaluator.py estén en el directorio actual.")
    sys.exit(1)

def extract_articles(texto):
    """
    Extrae artículos del texto usando expresiones regulares.
    Réplica de la función en prueba.py.
    """
    print("Extrayendo artículos del documento...")
    articulos = []
    
    # Buscar patrones "Artículo X."
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

def save_articles_by_cluster(articulos, pertenencia_articulos, categories, output_dir):
    """
    Guarda los artículos agrupados por cluster en archivos separados.
    
    Args:
        articulos: Lista de artículos
        pertenencia_articulos: Matriz de pertenencia de artículos a clusters
        categories: Lista de nombres de categorías
        output_dir: Directorio de salida
    
    Returns:
        Dictionary with cluster filenames as keys and paths as values
    """
    cluster_files = {}
    
    # Para cada categoría, guardar los 6 artículos con mayor pertenencia
    for cluster_id, categoria in enumerate(categories):
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
        nombre_archivo = f"cluster_{categoria_formato}_articulos.txt"
        ruta_archivo = os.path.join(output_dir, nombre_archivo)
        
        # Guardar artículos
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            f.write(f"Categoría: {categoria}\n")
            f.write(f"Top 6 artículos más relevantes:\n\n")
            
            # Guardar cada artículo con un separador claro y su grado de pertenencia
            for idx, (articulo, pertenencia) in enumerate(articulos_cluster, 1):
                f.write(f"\n\n{'='*50}\n")
                f.write(f"Artículo #{idx} en relevancia\n")
                f.write(f"{articulo['titulo']}\n")
                f.write(f"Grado de pertenencia: {pertenencia*100:.1f}%\n\n")
                f.write(articulo['contenido'])
        
        cluster_files[categoria] = ruta_archivo
        print(f"Guardado cluster '{categoria}' con {len(articulos_cluster)} artículos")
    
    return cluster_files

def read_file(input_file):
    """Lee un archivo y devuelve su contenido como texto."""
    try:
        # Comprobar si es un PDF o un archivo de texto
        if input_file.lower().endswith('.pdf'):
            try:
                # Intentar importar librería para leer PDFs
                from PyPDF2 import PdfReader
                reader = PdfReader(input_file)
                texto = ""
                for page in reader.pages:
                    texto += page.extract_text() + "\n"
                print(f"Archivo PDF leído: {len(texto)} caracteres")
                return texto
            except ImportError:
                print("No se pudo importar PyPDF2. Instálalo con: pip install PyPDF2")
                sys.exit(1)
        else:
            # Suponemos que es un archivo de texto
            with open(input_file, 'r', encoding='utf-8') as f:
                texto = f.read()
            print(f"Archivo de texto leído: {len(texto)} caracteres")
            return texto
    except Exception as e:
        print(f"Error leyendo el archivo {input_file}: {e}")
        return None

def main():
    # Configurar el analizador de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Pipeline completo de análisis, clustering y resumen de documentos de becas')
    
    # Rutas de entrada y salida
    parser.add_argument('--input', type=str, nargs='+', required=True, help='Archivos de entrada para procesar')
    parser.add_argument('--output', type=str, default='resultados_pipeline', help='Carpeta para guardar los resultados')
    
    # Opciones de topic modeling
    parser.add_argument('--num_topics', type=int, default=3, help='Número de temas para clustering')
    parser.add_argument('--custom_topics', type=str, nargs='+', 
                        help='Temas personalizados para buscar (ej: "requisitos_academicos")')
    
    # Opciones de resumen
    parser.add_argument('--summarization_models', type=str, nargs='+', default=['bart'], 
                       choices=['bart', 't5', 'longformer'], help='Modelos para resumir')
    
    # Opciones de evaluación
    parser.add_argument('--evaluate', action='store_true', help='Evaluar y comparar resúmenes con LLM')
    
    args = parser.parse_args()
    
    # Crear directorio de salida con marca de tiempo para evitar sobreescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear subdirectorios
    clusters_dir = os.path.join(output_dir, "clusters")
    summaries_dir = os.path.join(output_dir, "summaries")
    evaluations_dir = os.path.join(output_dir, "evaluations")
    
    os.makedirs(clusters_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(evaluations_dir, exist_ok=True)
    
    # Guardar configuración
    with open(os.path.join(output_dir, "pipeline_config.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Pipeline ejecutado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Archivos de entrada: {args.input}\n")
        f.write(f"Número de temas: {args.num_topics}\n")
        f.write(f"Temas personalizados: {args.custom_topics if args.custom_topics else 'No especificados'}\n")
        f.write(f"Modelos de resumen: {args.summarization_models}\n")
        f.write(f"Evaluación con LLM: {'Sí' if args.evaluate else 'No'}\n")
    
    # Inicializar modelos
    try:
        print("Inicializando modelo de topic modeling...")
        topic_model = BecasTransformerTopicModel()
        
        # Si hay temas personalizados, actualizar las categorías del modelo
        if args.custom_topics:
            custom_categories = {}
            for topic in args.custom_topics:
                if topic in topic_model.topic_categories:
                    custom_categories[topic] = topic_model.topic_categories[topic]
                else:
                    print(f"Advertencia: El tema personalizado '{topic}' no está predefinido.")
                    print(f"Temas disponibles: {list(topic_model.topic_categories.keys())}")
                    # Crear una categoría vacía que se poblará basándose en la similitud
                    custom_categories[topic] = []
            
            if custom_categories:
                topic_model.topic_categories = custom_categories
        
        # Usar las categorías del modelo
        categories = list(topic_model.topic_categories.keys())
        
        print("Inicializando modelo de resumen...")
        summarizer = ClusterSummarizer()
        
    except Exception as e:
        print(f"Error inicializando modelos: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Procesar cada archivo de entrada
    for input_file in args.input:
        try:
            print(f"\n{'='*60}")
            print(f"Procesando archivo: {input_file}")
            print(f"{'='*60}")
            file_basename = Path(input_file).stem
            
            # Crear directorios de salida específicos para este archivo
            file_clusters_dir = os.path.join(clusters_dir, file_basename)
            file_summaries_dir = os.path.join(summaries_dir, file_basename)
            
            os.makedirs(file_clusters_dir, exist_ok=True)
            os.makedirs(file_summaries_dir, exist_ok=True)
            
            # Leer el archivo de entrada
            texto_completo = read_file(input_file)
            if not texto_completo:
                print(f"No se pudo leer el archivo {input_file}. Pasando al siguiente archivo.")
                continue
            
            # Extraer artículos
            articulos = extract_articles(texto_completo)
            
            # Si no se encontraron artículos, pasar al siguiente archivo
            if not articulos:
                print(f"No se encontraron artículos en {input_file}. Pasando al siguiente archivo.")
                continue
            
            # Guardar todos los artículos en un solo archivo para referencia
            all_articles_file = os.path.join(file_clusters_dir, "todos_los_articulos.txt")
            with open(all_articles_file, 'w', encoding='utf-8') as f:
                f.write(f"Total de artículos extraídos: {len(articulos)}\n\n")
                
                for i, articulo in enumerate(articulos):
                    f.write(f"\n\n{'='*50}\n")
                    f.write(f"ARTÍCULO {i+1}: {articulo['titulo']}\n\n")
                    f.write(articulo['texto_completo'])
            
            # Cargar el documento en el modelo para cálculo de embeddings
            topic_model.raw_texts = [texto_completo]
            topic_model.doc_names = [file_basename]
            
            # Calcular embeddings del documento
            print("Calculando embeddings...")
            topic_model._compute_embeddings()
            
            # Extraer texto de cada artículo para cálculo de embeddings
            textos_articulos = [articulo['texto_completo'] for articulo in articulos]
            
            # Calcular embeddings para cada artículo
            print("Calculando embeddings para cada artículo...")
            if topic_model.model is not None:
                embeddings_articulos = topic_model.model.encode(textos_articulos)
                print("Embeddings calculados con modelo transformer")
            elif hasattr(topic_model, 'alternate_embedding_fn') and topic_model.alternate_embedding_fn is not None:
                embeddings_articulos = topic_model.alternate_embedding_fn(textos_articulos)
                print("Embeddings calculados con función alternativa")
            else:
                embeddings_articulos = topic_model._compute_tfidf_embeddings(textos_articulos)
                print("Embeddings calculados con TF-IDF")
            
            # Calcular similitud de cada artículo con cada categoría
            print("Calculando similitud de cada artículo con cada categoría...")
            from sklearn.metrics.pairwise import cosine_similarity
            
            n_articulos = len(articulos)
            n_categorias = len(categories)
            
            # Matriz para scores de similitud [n_articulos, n_categorias]
            similitud_matriz = np.zeros((n_articulos, n_categorias))
            
            for i, embedding_articulo in enumerate(embeddings_articulos):
                for j, categoria in enumerate(categories):
                    # Calcular similitud de coseno entre artículo y categoría
                    similitud = cosine_similarity([embedding_articulo], [topic_model.category_embeddings[categoria]])[0][0]
                    similitud_matriz[i, j] = similitud
            
            # Normalizar filas para que sumen 1 (convertir a "pertenencia difusa")
            suma_filas = similitud_matriz.sum(axis=1, keepdims=True)
            pertenencia_articulos = similitud_matriz / suma_filas
            
            # Guardar artículos por cluster
            print("Guardando artículos por cluster...")
            cluster_files = save_articles_by_cluster(articulos, pertenencia_articulos, categories, file_clusters_dir)
            
            # Resumir cada archivo de cluster
            print("\nGenerando resúmenes para cada cluster...")
            summary_files = {}
            
            for categoria, cluster_file in cluster_files.items():
                print(f"Procesando cluster: {categoria}")
                categoria_summaries_dir = os.path.join(file_summaries_dir, categoria.replace('_', '-'))
                os.makedirs(categoria_summaries_dir, exist_ok=True)
                
                # Generar resúmenes con los modelos solicitados
                model_summary_files = []
                for model_type in args.summarization_models:
                    print(f"  Generando resumen con modelo {model_type}...")
                    try:
                        # process_cluster creará un archivo de resumen
                        summary_file = summarizer.process_cluster(cluster_file, categoria_summaries_dir, model_type)
                        if summary_file:
                            model_summary_files.append(summary_file)
                            print(f"  Resumen guardado en: {os.path.basename(summary_file)}")
                    except Exception as e:
                        print(f"  Error generando resumen con {model_type}: {e}")
                        traceback.print_exc()
                
                summary_files[categoria] = model_summary_files
            
            # Evaluar resúmenes si se solicita y se usaron múltiples modelos
            if args.evaluate and len(args.summarization_models) > 1:
                print("\nEvaluando resúmenes con LLM...")
                file_evaluations_dir = os.path.join(evaluations_dir, file_basename)
                os.makedirs(file_evaluations_dir, exist_ok=True)
                
                for categoria, model_summaries in summary_files.items():
                    if len(model_summaries) > 1:
                        print(f"Evaluando resúmenes para categoría: {categoria}")
                        original_file = cluster_files[categoria]
                        output_file = os.path.join(file_evaluations_dir, f"{categoria.replace('_', '-')}_evaluacion.txt")
                        
                        try:
                            # Comparar resúmenes usando LLM
                            best_model, results = compare_summaries(original_file, model_summaries, output_file)
                            
                            # Imprimir resultados
                            print(f"  Mejor modelo: {best_model}")
                            for model, result in results.items():
                                print(f"  - {model}: {result['score']:.1f}/10")
                                
                            # Copiar el mejor resumen a un directorio 'best_summaries'
                            best_summaries_dir = os.path.join(output_dir, "best_summaries", file_basename)
                            os.makedirs(best_summaries_dir, exist_ok=True)
                            
                            # Encontrar el archivo del mejor resumen
                            best_summary_file = None
                            for summary_file in model_summaries:
                                if best_model in summary_file:
                                    best_summary_file = summary_file
                                    break
                            
                            if best_summary_file:
                                # Copiar al directorio de mejores resúmenes
                                dest_file = os.path.join(best_summaries_dir, f"{categoria.replace('_', '-')}_best.txt")
                                shutil.copy2(best_summary_file, dest_file)
                                print(f"  Mejor resumen copiado a: {os.path.relpath(dest_file, output_dir)}")
                        except Exception as e:
                            print(f"  Error evaluando resúmenes: {e}")
                            traceback.print_exc()
            
        except Exception as e:
            print(f"Error procesando archivo {input_file}: {e}")
            traceback.print_exc()
    
    print(f"\nPipeline completado. Resultados guardados en: {output_dir}")
    
    # Crear un informe simple resumiendo los resultados
    with open(os.path.join(output_dir, "resumen_pipeline.txt"), 'w', encoding='utf-8') as f:
        f.write(f"RESUMEN DE PIPELINE DE PROCESAMIENTO\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Archivos procesados: {len(args.input)}\n")
        f.write(f"Categorías analizadas: {categories}\n")
        f.write(f"Modelos de resumen utilizados: {args.summarization_models}\n\n")
        
        f.write(f"ESTRUCTURA DE RESULTADOS:\n")
        f.write(f"- clusters/: Contiene los artículos agrupados por categoría\n")
        f.write(f"- summaries/: Contiene los resúmenes generados por cada modelo\n")
        
        if args.evaluate and len(args.summarization_models) > 1:
            f.write(f"- evaluations/: Contiene las evaluaciones de los resúmenes\n")
            f.write(f"- best_summaries/: Contiene los mejores resúmenes según la evaluación\n")

if __name__ == "__main__":
    main()


    '''
    
    
    How to Use the Pipeline
Run the script with the following command structure:
bashCopiarpython pipeline.py --input archivo1.txt archivo2.txt 
                  --output resultados 
                  --summarization_models bart t5 
                  --evaluate
Command Line Arguments:

--input: One or more input files to process
--output: Base directory for storing results (a timestamp will be added automatically)
--num_topics: Number of topics for clustering (default: 3)
--custom_topics: Specify custom topics (e.g., "requisitos_academicos")
--summarization_models: Models to use for summarization (choices: bart, t5, longformer)
--evaluate: Flag to evaluate and compare summaries with LLM (only works when multiple models are specified)

'''
