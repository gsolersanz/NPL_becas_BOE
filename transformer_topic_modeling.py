#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm

# Verificar si el tokenizador de NLTK está disponible, descargarlo si no lo está
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class BecasTransformerTopicModel:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1'):
        """
        Inicializa el modelador de temas basado en transformers para documentos de becas.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers a utilizar
                        (por defecto: modelo multilingüe que funciona bien con español)
        """
        self.model_name = model_name
        
        # Inicializar modelo transformer
        print(f"Cargando modelo '{model_name}'...")
        try:
            # Intentar importar el módulo sentence_transformers aquí para mejor manejo de errores
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"Modelo cargado correctamente: {model_name}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Intentando usar una alternativa...")
            try:
                # Alternativa: usar un modelo más simple de HuggingFace
                from transformers import AutoTokenizer, AutoModel
                
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
                model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
                
                # Definir una función manual para obtener embeddings
                def get_embeddings(texts):
                    embeddings = []
                    for text in tqdm(texts, desc="Generando embeddings"):
                        # Truncar texto si es muy largo
                        if len(text) > 1000:
                            text = text[:1000]
                        
                        # Tokenizar y obtener embeddings
                        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        # Usar el embedding del token [CLS] como representación
                        embedding = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.append(embedding[0])
                    
                    return np.array(embeddings)
                
                # Guardar función alternativa
                self.model = None
                self.alternate_embedding_fn = get_embeddings
                print("Se usará un modelo alternativo basado en DistilBERT")
            except Exception as e:
                print(f"Error al cargar el modelo alternativo: {e}")
                print("Se usará un enfoque basado en frecuencia de palabras (más básico)")
                self.model = None
                self.alternate_embedding_fn = None
        
        # Definir las categorías de temas de interés con sus palabras clave
        self.topic_categories = {
            'documentación_y_plazos': [
                'documentación', 'entregar', 'presentar', 'solicitud', 'plazo', 
                'fecha límite', 'sede electrónica', 'registro', 'formulario', 
                'fecha de presentación', 'tramitación', 'pdf', 'dni', 'certificado',
                'acreditación', 'convocatoria', 'periodo', 'documento'
            ],
            'requisitos_económicos': [
                'renta', 'ingresos', 'umbral económico', 'patrimonio familiar', 
                'económico', 'deducciones', 'miembros computables', 'tributación',
                'declaración', 'impuesto', 'IRPF', 'recursos económicos', 'familia numerosa',
                'independencia económica', 'sustentador', 'patrimonio', 'valoración'
            ],
            'requisitos_académicos': [
                'expediente académico', 'créditos', 'asignaturas', 'matriculación', 
                'nota media', 'calificaciones', 'rendimiento académico', 'mínimo', 
                'curso completo', 'evaluación', 'cualificación', 'estudios',
                'titulación', 'graduado', 'bachillerato', 'universidad', 'máster'
            ],
            'cuantías_y_ayudas': [
                'cuantía', 'importe', 'beca', 'ayuda', 'euros', 'variable', 'fija',
                'compensación', 'matrícula', 'alojamiento', 'residencia', 'material',
                'desplazamiento', 'componentes', 'cantidad', 'percibir', 'pago'
            ],
            'procedimiento_resolución': [
                'procedimiento', 'resolución', 'notificación', 'recurso', 'concesión', 
                'denegación', 'alegaciones', 'reclamación', 'subsanación', 
                'requerimientos', 'evaluación', 'criterios', 'publicación', 'listado',
                'comisión', 'selección', 'adjudicación', 'incompatibilidad', 'reintegro'
            ]
        }
        
        # Variables para almacenar datos
        self.raw_texts = []
        self.doc_names = []
        self.paragraphs = []
        self.paragraph_doc_ids = []
        self.embeddings = None
        self.paragraph_embeddings = None
        self.topic_centers = None
        self.category_embeddings = None
    
    def extract_by_category(self, text, category_name):
        """
        Extrae partes relevantes del texto relacionadas con una categoría específica.
        
        Args:
            text: Texto a analizar
            category_name: Nombre de la categoría a buscar
            
        Returns:
            Texto con las secciones relevantes
        """
        # Verificar si la categoría existe
        if category_name not in self.topic_categories:
            return f"Error: La categoría '{category_name}' no existe."
        
        # Dividir texto en párrafos
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Obtener palabras clave de la categoría
        keywords = self.topic_categories[category_name]
        
        # Calcular relevancia de cada párrafo
        relevant_paragraphs = []
        
        for para in paragraphs:
            para_lower = para.lower()
            relevance_score = 0
            
            # Contar ocurrencias de palabras clave
            for keyword in keywords:
                if keyword.lower() in para_lower:
                    relevance_score += 1
            
            if relevance_score > 0:
                relevant_paragraphs.append((para, relevance_score))
        
        # Ordenar por relevancia
        relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar los más relevantes
        top_paragraphs = [p[0] for p in relevant_paragraphs[:10]]
        
        # Si no se encontraron párrafos relevantes
        if not top_paragraphs:
            return f"No se encontraron secciones relevantes para la categoría '{category_name}'."
        
        # Unir y devolver
        return "\n\n".join(top_paragraphs)
    
    def _preprocess_documents(self, max_length=512, min_length=50):
        """
        Preprocesa los documentos dividiéndolos en párrafos para el análisis.
        
        Args:
            max_length: Longitud máxima de un párrafo (para control de memoria)
            min_length: Longitud mínima de un párrafo para ser considerado
        """
        self.paragraphs = []
        self.paragraph_doc_ids = []
        
        for doc_id, text in enumerate(self.raw_texts):
            # Dividir en párrafos
            doc_paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            # Para párrafos muy largos, dividirlos en fragmentos más pequeños
            processed_paragraphs = []
            for para in doc_paragraphs:
                if len(para) > max_length:
                    # Intentar dividir por oraciones
                    sentences = sent_tokenize(para)
                    current_para = ""
                    
                    for sent in sentences:
                        if len(current_para) + len(sent) < max_length:
                            current_para += " " + sent if current_para else sent
                        else:
                            if len(current_para) >= min_length:
                                processed_paragraphs.append(current_para)
                            current_para = sent
                    
                    if current_para and len(current_para) >= min_length:
                        processed_paragraphs.append(current_para)
                elif len(para) >= min_length:
                    processed_paragraphs.append(para)
            
            # Añadir a la lista global
            for para in processed_paragraphs:
                self.paragraphs.append(para)
                self.paragraph_doc_ids.append(doc_id)
        
        print(f"Documentos procesados en {len(self.paragraphs)} párrafos.")
    
    def _compute_embeddings(self, batch_size=32):
        """
        Calcula los embeddings para cada documento y párrafo.
        
        Args:
            batch_size: Tamaño del lote para procesar los embeddings
        """
        # Calcular embeddings para documentos completos
        print("Calculando embeddings para documentos completos...")
        
        if self.model is not None:
            # Usar SentenceTransformer
            self.embeddings = self.model.encode(self.raw_texts, batch_size=batch_size, 
                                               show_progress_bar=True)
        elif self.alternate_embedding_fn is not None:
            # Usar función alternativa
            self.embeddings = self.alternate_embedding_fn(self.raw_texts)
        else:
            # Usar enfoque basado en frecuencia de palabras
            self.embeddings = self._compute_tfidf_embeddings(self.raw_texts)
        
        # Preprocesar documentos en párrafos
        self._preprocess_documents()
        
        # Calcular embeddings para párrafos
        print("Calculando embeddings para párrafos...")
        if self.model is not None:
            # Usar SentenceTransformer
            self.paragraph_embeddings = self.model.encode(self.paragraphs, batch_size=batch_size, 
                                                        show_progress_bar=True)
        elif self.alternate_embedding_fn is not None:
            # Usar función alternativa
            self.paragraph_embeddings = self.alternate_embedding_fn(self.paragraphs)
        else:
            # Usar enfoque basado en frecuencia de palabras
            self.paragraph_embeddings = self._compute_tfidf_embeddings(self.paragraphs)
        
        # Calcular embeddings para las categorías definidas
        self.category_embeddings = {}
        print("Calculando embeddings para categorías predefinidas...")
        for category, keywords in self.topic_categories.items():
            if self.model is not None:
                # Calcular embedding para cada palabra clave
                keyword_embeddings = self.model.encode(keywords, batch_size=batch_size, 
                                                     show_progress_bar=False)
                # Calcular el embedding promedio para la categoría
                self.category_embeddings[category] = np.mean(keyword_embeddings, axis=0)
            elif self.alternate_embedding_fn is not None:
                # Usar función alternativa
                keyword_embeddings = self.alternate_embedding_fn(keywords)
                self.category_embeddings[category] = np.mean(keyword_embeddings, axis=0)
            else:
                # Usar enfoque basado en frecuencia de palabras
                self.category_embeddings[category] = self._compute_tfidf_embeddings([" ".join(keywords)])[0]
    
    def _compute_tfidf_embeddings(self, texts):
        """
        Calcula embeddings simples basados en frecuencia de palabras (TF-IDF).
        Este método se usa como último recurso si fallan los modelos más avanzados.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("Usando embeddings basados en TF-IDF (método alternativo)...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            X = vectorizer.fit_transform(texts)
            return X.toarray()
        except:
            # Si hay algún problema, devolver vectores aleatorios como último recurso
            print("Error en TF-IDF. Usando vectores aleatorios como último recurso.")
            return np.random.rand(len(texts), 300)
    
    def _cluster_paragraphs(self, n_clusters=5):
        """
        Agrupa los párrafos en clusters para identificar temas.
        
        Args:
            n_clusters: Número de clusters (temas) a identificar
        """
        print(f"Agrupando párrafos en {n_clusters} clusters...")
        
        # Aplicar KMeans para agrupar párrafos
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.paragraph_clusters = kmeans.fit_predict(self.paragraph_embeddings)
        self.topic_centers = kmeans.cluster_centers_
        
        # Mapear clusters a categorías predefinidas
        self.cluster_to_category = {}
        
        for cluster_id in range(n_clusters):
            similarities = {}
            for category, embedding in self.category_embeddings.items():
                similarity = cosine_similarity([self.topic_centers[cluster_id]], [embedding])[0][0]
                similarities[category] = similarity
            
            # Asignar el cluster a la categoría más similar
            best_category = max(similarities.items(), key=lambda x: x[1])
            self.cluster_to_category[cluster_id] = best_category[0]
            
            print(f"Cluster {cluster_id} mapeado a '{best_category[0]}' (similitud: {best_category[1]:.4f})")
        
        # Crear atributo con párrafos por cluster
        self.paragraphs_by_cluster = {i: [] for i in range(n_clusters)}
        for idx, cluster in enumerate(self.paragraph_clusters):
            self.paragraphs_by_cluster[cluster].append(idx)
    
    def _analyze_documents(self):
        """
        Analiza los documentos para determinar la distribución de temas.
        """
        # Matriz de distribución de temas por documento
        n_docs = len(self.raw_texts)
        n_clusters = len(self.topic_centers)
        self.topic_distribution = np.zeros((n_docs, n_clusters))
        
        # Contar ocurrencias de cada cluster por documento
        for para_idx, cluster in enumerate(self.paragraph_clusters):
            doc_id = self.paragraph_doc_ids[para_idx]
            self.topic_distribution[doc_id, cluster] += 1
        
        # Normalizar para obtener porcentajes
        row_sums = self.topic_distribution.sum(axis=1, keepdims=True)
        self.topic_distribution = self.topic_distribution / row_sums
        
        # Crear DataFrame con la distribución
        topics = [self.cluster_to_category[i] for i in range(n_clusters)]
        self.topic_df = pd.DataFrame(self.topic_distribution, index=self.doc_names, columns=topics)
    
    def analyze_documents(self, n_clusters=5):
        """
        Realiza el análisis completo de los documentos.
        
        Args:
            n_clusters: Número de clusters (temas) a identificar
            
        Returns:
            DataFrame con los resultados del análisis
        """
        # Calcular embeddings
        self._compute_embeddings()
        
        # Agrupar párrafos
        self._cluster_paragraphs(n_clusters)
        
        # Analizar distribución de temas
        self._analyze_documents()
        
        # Crear DataFrame de resultados
        results = []
        
        for doc_id, doc_name in enumerate(self.doc_names):
            # Obtener distribución de temas para este documento
            topic_dist = self.topic_df.iloc[doc_id].to_dict()
            
            # Determinar tema principal
            main_topic = max(topic_dist.items(), key=lambda x: x[1])[0]
            
            # Extraer fragmentos relevantes para el tema principal
            relevant_text = self.extract_relevant_sections(doc_id, main_topic)
            
            results.append({
                'documento': doc_name,
                'tema_principal': main_topic,
                'distribución_temas': topic_dist,
                'texto_relevante': relevant_text[:1500] + "..." if len(relevant_text) > 1500 else relevant_text
            })
        
        return pd.DataFrame(results)
    
    def extract_relevant_sections(self, doc_id, topic, top_n=10):
        """
        Extrae las secciones más relevantes de un documento para un tema específico.
        
        Args:
            doc_id: ID del documento
            topic: Nombre del tema
            top_n: Número de párrafos relevantes a extraer
            
        Returns:
            Texto con las secciones relevantes
        """
        # Encontrar el cluster correspondiente al tema
        cluster_id = None
        for cluster, category in self.cluster_to_category.items():
            if category == topic:
                cluster_id = cluster
                break
        
        if cluster_id is None:
            return "No se encontraron secciones relevantes para este tema."
        
        # Obtener los índices de párrafos de este documento
        doc_paragraph_indices = [i for i, doc in enumerate(self.paragraph_doc_ids) if doc == doc_id]
        
        # Filtrar solo los párrafos de este cluster
        relevant_indices = [i for i in doc_paragraph_indices 
                        if self.paragraph_clusters[i] == cluster_id]
        
        # Si no hay párrafos del cluster, buscar los más cercanos al centro del tema
        if not relevant_indices:
            # Calcular similitud de cada párrafo con el centro del tema
            similarities = []
            for i in doc_paragraph_indices:
                similarity = cosine_similarity([self.paragraph_embeddings[i]], 
                                            [self.topic_centers[cluster_id]])[0][0]
                similarities.append((i, similarity))
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Tomar los top_n párrafos más similares
            relevant_indices = [idx for idx, _ in similarities[:top_n]]
        
        # Ordenar por posición en el documento
        relevant_indices.sort()
        
        # Extraer y unir párrafos relevantes sin añadir los marcadores [n.m]
        selected_paragraphs = [self.paragraphs[i] for i in relevant_indices]
        
        return "\n\n".join(selected_paragraphs)
    
    def visualize_topics(self, output_path=None):
        """
        Crea una visualización de la distribución de temas.
        
        Args:
            output_path: Ruta para guardar la visualización
        """
        # Crear heatmap
        plt.figure(figsize=(12, max(8, len(self.doc_names) * 0.4)))
        sns.heatmap(self.topic_df, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".2f")
        plt.title("Distribución de Temas por Documento")
        plt.ylabel("Documentos")
        plt.xlabel("Temas")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Visualización guardada en {output_path}")
        else:
            plt.show()
    
    def save_topic_sections(self, output_folder):
        """
        Guarda las secciones relevantes de cada documento por tema.
        
        Args:
            output_folder: Carpeta donde guardar los resultados
        """
        os.makedirs(output_folder, exist_ok=True)
        
        for doc_id, doc_name in enumerate(self.doc_names):
            # Obtener distribución de temas para este documento
            topic_dist = self.topic_df.iloc[doc_id].to_dict()
            
            # Para cada tema con probabilidad significativa
            for topic, prob in topic_dist.items():
                if prob > 0.1:  # Solo considerar temas con probabilidad significativa
                    relevant_text = self.extract_relevant_sections(doc_id, topic)
                    
                    if relevant_text:
                        # Crear nombre de archivo
                        safe_topic = topic.replace('_', '-')
                        filename = f"{doc_name}_{safe_topic}.txt"
                        filepath = os.path.join(output_folder, filename)
                        
                        # Guardar el texto
                        with open(filepath, 'w', encoding='utf-8') as file:
                            file.write(f"Documento: {doc_name}\n")
                            file.write(f"Tema: {topic}\n")
                            file.write(f"Probabilidad: {prob:.4f}\n\n")
                            file.write(relevant_text)
        
        print(f"Secciones temáticas guardadas en {output_folder}")
    
    def search_by_query(self, query, top_n=5):
        """
        Busca los párrafos más relevantes para una consulta.
        
        Args:
            query: Texto de la consulta
            top_n: Número de resultados a devolver
            
        Returns:
            Lista de tuplas (documento, párrafo, puntuación)
        """
        # Verificar si se han calculado embeddings
        if self.paragraph_embeddings is None:
            raise ValueError("Primero debes ejecutar analyze_documents")
        
        # Convertir la consulta a embedding
        if self.model is not None:
            query_embedding = self.model.encode([query])[0]
        elif self.alternate_embedding_fn is not None:
            query_embedding = self.alternate_embedding_fn([query])[0]
        else:
            query_embedding = self._compute_tfidf_embeddings([query])[0]
        
        # Calcular similitud con todos los párrafos
        similarities = cosine_similarity([query_embedding], self.paragraph_embeddings)[0]
        
        # Obtener los índices de los top_n párrafos más similares
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # Crear resultados
        results = []
        for idx in top_indices:
            doc_id = self.paragraph_doc_ids[idx]
            doc_name = self.doc_names[doc_id]
            paragraph = self.paragraphs[idx]
            score = similarities[idx]
            
            results.append({
                'documento': doc_name,
                'párrafo': paragraph,
                'puntuación': score
            })
        
        return pd.DataFrame(results)