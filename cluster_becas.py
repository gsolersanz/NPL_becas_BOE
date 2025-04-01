#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script mejorado para dividir documentos de becas en artículos, agruparlos en 3 clusters
y guardar los artículos de cada cluster en archivos separados.
"""

import os
import re
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def extraer_articulos(texto):
    """
    Extrae artículos del texto utilizando un patrón específico para documentos de becas.
    
    Args:
        texto: Texto completo del documento
        
    Returns:
        Lista de diccionarios con los artículos extraídos (título, contenido, texto completo)
    """
    # Patrón mejorado para capturar artículos en el formato "Artículo X. Título."
    articulos = []
    
    # Buscar todos los "Artículo N." en el texto
    inicio_articulos = re.finditer(r'Artículo\s+\d+\.[\s\S]*?(?=Artículo\s+\d+\.|$)', texto, re.MULTILINE)
    
    for match in inicio_articulos:
        texto_articulo = match.group(0).strip()
        
        # Extraer número y título del artículo
        info_match = re.match(r'Artículo\s+(\d+)\.[\s]*(.*?)\.', texto_articulo)
        
        if info_match:
            num_articulo = info_match.group(1)
            titulo_articulo = info_match.group(2).strip()
            
            # Si el título está vacío o no se encontró, usar un texto genérico
            if not titulo_articulo:
                titulo_articulo = f"Artículo {num_articulo}"
            
            # El contenido es todo lo que viene después del título
            titulo_completo = f"Artículo {num_articulo}. {titulo_articulo}."
            inicio_contenido = texto_articulo.find(titulo_completo) + len(titulo_completo)
            contenido = texto_articulo[inicio_contenido:].strip()
            
            articulo = {
                'numero': num_articulo,
                'titulo': titulo_articulo,
                'titulo_completo': titulo_completo,
                'contenido': contenido,
                'texto_completo': texto_articulo
            }
            
            articulos.append(articulo)
        else:
            # Si no se pudo extraer correctamente título y número, guardar el texto completo
            articulo = {
                'numero': 'N/A',
                'titulo': 'Sin título',
                'titulo_completo': 'Artículo sin título identificado',
                'contenido': texto_articulo,
                'texto_completo': texto_articulo
            }
            articulos.append(articulo)
    
    # Si no se encontraron artículos, probar con un enfoque alternativo
    if not articulos:
        # Dividir por capítulos o secciones
        capitulos = re.finditer(r'CAPÍTULO\s+[IVX]+\.[\s\S]*?(?=CAPÍTULO\s+[IVX]+\.|$)', texto, re.MULTILINE)
        
        for i, match in enumerate(capitulos):
            texto_capitulo = match.group(0).strip()
            
            # Extraer número de capítulo
            info_match = re.match(r'CAPÍTULO\s+([IVX]+)\.[\s]*(.*?)\.', texto_capitulo)
            
            if info_match:
                num_capitulo = info_match.group(1)
                titulo_capitulo = info_match.group(2).strip()
                
                articulo = {
                    'numero': f"Cap. {num_capitulo}",
                    'titulo': titulo_capitulo,
                    'titulo_completo': f"CAPÍTULO {num_capitulo}. {titulo_capitulo}.",
                    'contenido': texto_capitulo,
                    'texto_completo': texto_capitulo
                }
            else:
                articulo = {
                    'numero': f"Cap. {i+1}",
                    'titulo': 'Sin título',
                    'titulo_completo': f"CAPÍTULO sin título identificado",
                    'contenido': texto_capitulo,
                    'texto_completo': texto_capitulo
                }
            
            articulos.append(articulo)
    
    # Si todavía no hay resultados, dividir por secciones significativas
    if not articulos:
        # Buscar secciones significativas
        sections = []
        
        # Lista de patrones para buscar secciones
        section_patterns = [
            r'CAPÍTULO\s+[IVX]+\.\s*(.*?)(?=\n)',
            r'SECCIÓN\s+\d+ª\.\s*(.*?)(?=\n)',
            r'TÍTULO\s+[IVX]+\.\s*(.*?)(?=\n)',
            r'(\d+\.\d+)\.\s*(.*?)(?=\n)',
            r'(\d+\.)\s*[A-ZÁÉÍÓÚÑ].*?(?=\n)',
            r'[A-Z]{2,}.*?(?=\n)'
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, texto, re.MULTILINE)
            sections.extend([(m.start(), m.group(0)) for m in matches])
        
        # Ordenar secciones por posición
        sections.sort(key=lambda x: x[0])
        
        # Extraer contenido entre secciones
        if sections:
            for i in range(len(sections)):
                start_pos = sections[i][0]
                title = sections[i][1]
                
                # Determinar dónde termina esta sección
                if i < len(sections) - 1:
                    end_pos = sections[i+1][0]
                else:
                    end_pos = len(texto)
                
                section_text = texto[start_pos:end_pos].strip()
                
                if len(section_text) > 100:  # Solo secciones significativas
                    articulo = {
                        'numero': str(i+1),
                        'titulo': title,
                        'titulo_completo': title,
                        'contenido': section_text,
                        'texto_completo': section_text
                    }
                    articulos.append(articulo)
    
    # Si aún no hay resultados, dividir por párrafos
    if not articulos:
        parrafos = [p for p in re.split(r'\n\s*\n', texto) if len(p.strip()) > 100]
        
        for i, parrafo in enumerate(parrafos):
            articulo = {
                'numero': str(i+1),
                'titulo': f"Párrafo {i+1}",
                'titulo_completo': f"Párrafo {i+1}",
                'contenido': parrafo,
                'texto_completo': parrafo
            }
            articulos.append(articulo)
    
    # Filtrar para eliminar contenido no relevante como direcciones de validación repetidas
    articulos_filtrados = []
    textos_vistos = set()
    
    for articulo in articulos:
        # Obtener una versión simplificada del texto para comparación
        texto_simple = re.sub(r'\s+', ' ', articulo['texto_completo']).strip()
        
        # Verificar si no es contenido repetitivo
        if not "DIRECCIÓN DE VALIDACIÓN" in texto_simple and texto_simple not in textos_vistos:
            # Verificar si tiene contenido sustancial
            if len(texto_simple) > 100 and len(texto_simple.split()) > 20:
                articulos_filtrados.append(articulo)
                textos_vistos.add(texto_simple)
    
    return articulos_filtrados

def main():
    """Función principal para procesar documentos de becas."""
    # Importar el modelo de topic modeling
    try:
        from transformer_topic_modeling import BecasTransformerTopicModel
        print("Modelo BecasTransformerTopicModel importado correctamente")
    except ImportError as e:
        print(f"Error al importar BecasTransformerTopicModel: {e}")
        print("Asegúrate de que el archivo transformer_topic_modeling.py está en el mismo directorio")
        return
    
    # Directorio de salida
    output_dir = os.path.join(os.getcwd(), "resultados_clusters")
    os.makedirs(output_dir, exist_ok=True)
    
    # Archivo a procesar (cambiar según sea necesario)
    txt_file = "corpus_txt/ayudas_21-22.txt"
    if not os.path.exists(txt_file):
        txt_file = input("Introduce la ruta al archivo de texto/PDF a procesar: ")
    
    # Si el archivo es PDF, convertirlo a texto
    if txt_file.lower().endswith('.pdf'):
        try:
            from pdf_totext import pdf_to_text
            print(f"Convirtiendo PDF a texto: {txt_file}")
            pdf_folder = os.path.dirname(txt_file) or "."
            txt_folder = os.path.join(os.getcwd(), "corpus_txt")
            os.makedirs(txt_folder, exist_ok=True)
            pdf_to_text(pdf_folder, txt_folder)
            txt_file = os.path.join(txt_folder, Path(txt_file).stem + ".txt")
        except Exception as e:
            print(f"Error al convertir PDF a texto: {e}")
            return
    
    # Leer el archivo de texto
    try:
        with open(txt_file, 'r', encoding='utf-8') as file:
            texto = file.read()
        print(f"Archivo leído correctamente: {txt_file} ({len(texto)} caracteres)")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return
    
    # Extraer artículos
    print("Extrayendo artículos del documento...")
    articulos = extraer_articulos(texto)
    print(f"Se han extraído {len(articulos)} artículos")
    
    if not articulos:
        print("No se pudieron extraer artículos del documento.")
        return
    
    # Mostrar los primeros artículos extraídos
    print("\nPrimeros 2 artículos extraídos:")
    for i, articulo in enumerate(articulos[:2]):
        print(f"Artículo {articulo['numero']}: {articulo['titulo']}")
        print(f"Longitud: {len(articulo['texto_completo'])} caracteres")
        print(f"Primeras 100 caracteres: {articulo['texto_completo'][:100]}...")
        print("-" * 50)
    
    # Inicializar el modelo
    print("Inicializando modelo de análisis de temas...")
    modeler = BecasTransformerTopicModel()
    
    # Añadir el documento
    modeler.raw_texts = [texto]
    modeler.doc_names = [Path(txt_file).stem]
    
    # Definir tres categorías principales para los documentos de becas
    modeler.topic_categories = {
        'requisitos_academicos': [
            'nota media', 'calificaciones', 'créditos', 'evaluación', 
            'curso completo', 'rendimiento', 'graduado', 'universidad', 
            'bachillerato', 'máster', 'matrícula', 'asignaturas',
            'expediente', 'titulación', 'superado', 'aprobar', 'grado',
            'enseñanzas', 'módulos', 'plan de estudios', 'estudiantes'
        ],
        'requisitos_economicos': [
            'renta', 'ingresos', 'patrimonio', 'umbral', 
            'cuantía', 'deducciones', 'recursos económicos', 
            'compensación', 'tributación', 'familia numerosa', 
            'miembros computables', 'patrimonio familiar', 'ayudas',
            'euros', 'importe', 'beca de matrícula', 'beneficiarios',
            'perciban', 'financiación', 'presupuestos'
        ],
        'procedimiento_y_plazos': [
            'solicitud', 'convocatoria', 'plazo', 'fecha límite', 
            'documentación', 'inscripción', 'registro', 'presentación', 
            'tramitación', 'sede electrónica', 'certificado', 
            'lugar de entrega', 'requisitos', 'procedimiento', 
            'notificación', 'resolución', 'recurso', 'reintegro',
            'presentar', 'información', 'impreso', 'verificación'
        ]
    }
    
    # Realizar análisis de temas con 3 clusters
    print("Computando embeddings...")
    modeler._compute_embeddings()
    
    print("Agrupando en 3 clusters...")
    modeler._cluster_paragraphs(n_clusters=3)
    
    # Extraer textos de artículos para calcular embeddings
    article_texts = [articulo['texto_completo'] for articulo in articulos]
    
    # Calcular embeddings para cada artículo
    print("Calculando embeddings para cada artículo...")
    
    if modeler.model is not None:
        article_embeddings = modeler.model.encode(article_texts, show_progress_bar=True)
    elif hasattr(modeler, 'alternate_embedding_fn') and modeler.alternate_embedding_fn is not None:
        article_embeddings = modeler.alternate_embedding_fn(article_texts)
    else:
        article_embeddings = modeler._compute_tfidf_embeddings(article_texts)
    
    # Asignar cada artículo a un cluster basado en similitud
    print("Asignando artículos a clusters...")
    article_clusters = []
    
    for embedding in tqdm(article_embeddings, desc="Asignando clusters"):
        # Calcular similitud con cada centro de cluster
        similarities = cosine_similarity([embedding], modeler.topic_centers)[0]
        # Asignar al cluster más similar
        cluster_id = np.argmax(similarities)
        article_clusters.append(cluster_id)
    
    # Organizar artículos por cluster
    cluster_articles = {i: [] for i in range(3)}
    
    for idx, cluster_id in enumerate(article_clusters):
        cluster_articles[cluster_id].append(articulos[idx])
    
    # Guardar artículos de cada cluster en archivos separados
    print("Guardando artículos por cluster...")
    
    for cluster_id, cluster_content in cluster_articles.items():
        # Obtener la categoría del cluster
        category = modeler.cluster_to_category.get(cluster_id, f"cluster_{cluster_id}")
        
        # Crear nombre de archivo
        safe_category = category.replace('_', '-')
        filename = f"{Path(txt_file).stem}_{safe_category}_articulos.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Guardar artículos
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f"Documento: {Path(txt_file).stem}\n")
            file.write(f"Categoría: {category}\n")
            file.write(f"Número de artículos: {len(cluster_content)}\n\n")
            
            # Guardar cada artículo con un separador claro
            for articulo in cluster_content:
                file.write(f"\n\n{'='*50}\n")
                file.write(f"{articulo['titulo_completo']}\n\n")
                file.write(articulo['contenido'])
        
        print(f"Cluster '{category}' guardado en {filename} con {len(cluster_content)} artículos")
    
    print(f"\nProceso completado. Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()