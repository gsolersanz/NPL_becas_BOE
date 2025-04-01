#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path
import torch
import re
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    LongformerTokenizer, LongformerForSequenceClassification,
    LEDForConditionalGeneration, LEDTokenizer
)
from tqdm import tqdm
import huggingface_hub
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 30  # Aumentar a 30 segundos

class ClusterSummarizer:
    def __init__(self, device=None):
        """
        Inicializa el resumidor de clusters de artículos.
        
        Args:
            device: Dispositivo a utilizar para los modelos (cuda o cpu)
        """
        # Determinar dispositivo (GPU o CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilizando dispositivo: {self.device}")
        
        # Inicializar los modelos
        self.models = {
            'bart': {
                'name': 'facebook/bart-base',  # Modelo más pequeño
                'model': None,
                'tokenizer': None,
                'max_length': 1024,
                'min_length': 50,
                'length_penalty': 2.0,
                'num_beams': 4
            },
            't5': {
                'name': 't5-small',  # Modelo más pequeño
                'model': None,
                'tokenizer': None,
                'max_length': 512,
                'min_length': 50,
                'length_penalty': 2.0,
                'num_beams': 4
            },
            'longformer': {
                'name': 'allenai/led-base-16384',  # LED (Longformer Encoder-Decoder) para resumen
                'model': None,
                'tokenizer': None,
                'max_length': 4096,  # Puede manejar secuencias mucho más largas
                'min_length': 100,
                'length_penalty': 2.0,
                'num_beams': 4,
                'global_attention_indices': [0]  # Atención global para el token [CLS]
            }
        }
    
    def load_model(self, model_type):
        """
        Carga un modelo específico bajo demanda.
        
        Args:
            model_type: Tipo de modelo ('bart', 't5', 'pegasus')
        """
        if model_type not in self.models:
            raise ValueError(f"Modelo no soportado: {model_type}")
        
        config = self.models[model_type]
        
        if config['model'] is None:
            print(f"Cargando modelo {model_type} ({config['name']})...")
            
            # Cargar según el tipo de modelo
            if model_type == 'bart':
                config['tokenizer'] = BartTokenizer.from_pretrained(config['name'])
                config['model'] = BartForConditionalGeneration.from_pretrained(config['name']).to(self.device)
            
            elif model_type == 't5':
                config['tokenizer'] = T5Tokenizer.from_pretrained(config['name'])
                config['model'] = T5ForConditionalGeneration.from_pretrained(config['name']).to(self.device)
            
            elif model_type == 'pegasus':
                config['tokenizer'] = PegasusTokenizer.from_pretrained(config['name'])
                config['model'] = PegasusForConditionalGeneration.from_pretrained(config['name']).to(self.device)
            
            elif model_type == 'longformer':
                config['tokenizer'] = LEDTokenizer.from_pretrained(config['name'])
                config['model'] = LEDForConditionalGeneration.from_pretrained(config['name']).to(self.device)
            
            print(f"Modelo {model_type} cargado correctamente.")
    
    def generate_summary(self, text, model_type):
        """
        Genera un resumen utilizando el modelo especificado.
        
        Args:
            text: Texto a resumir
            model_type: Tipo de modelo a utilizar
            
        Returns:
            Resumen generado
        """
        # Cargar el modelo si aún no se ha cargado
        self.load_model(model_type)
        
        config = self.models[model_type]
        model = config['model']
        tokenizer = config['tokenizer']
        
        # Preparar el texto según el tipo de modelo
        if model_type == 't5':
            text_prepared = f"summarize: {text}"
        else:
            text_prepared = text
            
        # Para Longformer, podríamos necesitar un manejo especial
        if model_type == 'longformer':
            # Longformer puede manejar textos más largos, así que no es necesario truncar tanto
            max_input_length = config['max_length']
        
        # Truncar texto si es muy largo
        max_input_length = min(config['max_length'], 1024)
        if len(text_prepared) > max_input_length * 4:
            print(f"Texto demasiado largo, truncando a {max_input_length*4} caracteres")
            text_prepared = text_prepared[:max_input_length*2] + " ... " + text_prepared[-max_input_length*2:]
        
        try:
            # Tokenizar el texto
            if model_type == 'longformer':
                # Configuración especial para Longformer/LED
                inputs = tokenizer(text_prepared, max_length=max_input_length, 
                               truncation=True, padding="longest", return_tensors="pt")
                
                # Si estamos usando LED, podemos configurar la atención global para ciertos tokens
                if hasattr(config, 'global_attention_indices'):
                    # Configurar atención global para los tokens especificados (típicamente el token CLS/BOS)
                    global_attention_mask = torch.zeros_like(inputs['input_ids'])
                    for idx in config['global_attention_indices']:
                        if idx < global_attention_mask.shape[1]:
                            global_attention_mask[:, idx] = 1
                    inputs['global_attention_mask'] = global_attention_mask
            else:
                # Tokenización estándar para otros modelos
                inputs = tokenizer(text_prepared, max_length=max_input_length, 
                               truncation=True, padding="longest", return_tensors="pt")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generar resumen
            try:
                summary_ids = model.generate(
                    **inputs,
                    max_length=min(config['max_length'], 512),
                    min_length=min(150, max_input_length // 4),
                    length_penalty=1.5,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                
                # Decodificar el resumen
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
            except Exception as e:
                print(f"Error generando resumen con {model_type}: {e}")
                # Si falla, intentar con una configuración más simple
                try:
                    print("Intentando configuración alternativa...")
                    summary_ids = model.generate(
                        **inputs,
                        max_length=256,
                        min_length=50,
                        num_beams=2,
                        early_stopping=True
                    )
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                except:
                    summary = "Error al generar resumen. El texto puede ser demasiado complejo."
            
            return summary
        
        except Exception as e:
            print(f"Error en la preparación del resumen con {model_type}: {e}")
            return f"Error de preparación: {str(e)}"
    
    def split_into_articles(self, text):
        """
        Divide el texto del cluster en artículos individuales.
        
        Args:
            text: Texto completo del cluster
            
        Returns:
            Tuple con (header, processed_articles)
        """
        # Dividir el texto por el separador estándar de artículos
        articles_sections = re.split(r'={10,}\s*\n', text)
        
        # Filtrar secciones vacías y eliminar espacios en blanco
        articles_sections = [section.strip() for section in articles_sections if section.strip()]
        
        # El primer fragmento es el encabezado
        header = articles_sections[0] if articles_sections else ""
        content_articles = articles_sections[1:] if len(articles_sections) > 1 else []
        
        # Extraer información relevante de cada artículo
        processed_articles = []
        
        for article in content_articles:
            lines = article.split('\n')
            
            # Verificar si el formato es consistente con un artículo
            if len(lines) >= 2:
                # Buscar el número de artículo y la relevancia
                article_num_match = re.search(r'Artículo #(\d+) en relevancia', lines[0])
                article_id_match = re.search(r'Artículo (\d+)\.', lines[1] if len(lines) > 1 else "")
                relevance_match = re.search(r'Grado de pertenencia: ([\d.]+)%', lines[2] if len(lines) > 2 else "")
                
                if article_num_match:
                    article_num = article_id_match.group(1) if article_id_match else article_num_match.group(1)
                    relevance = float(relevance_match.group(1)) if relevance_match else 0.0
                    
                    # Buscar el título completo del artículo y su contenido
                    article_title = ""
                    article_content = ""
                    title_found = False
                    
                    for i in range(3, len(lines)):
                        line = lines[i].strip()
                        # Buscar la línea que comienza con "Artículo N." como título completo
                        if not title_found and line.startswith('Artículo'):
                            article_title = line
                            title_found = True
                        # Todo lo demás es contenido
                        elif i > 3:  # Asegurarse de que no capturamos líneas en blanco al principio
                            article_content += lines[i] + "\n"
                    
                    # Si no encontramos un título formateado como esperábamos, usar la segunda línea
                    if not article_title and article_id_match:
                        article_title = lines[1].strip()
                    
                    # Si encontramos un artículo bien formado
                    processed_articles.append({
                        'number': article_num,
                        'relevance': relevance,
                        'title': article_title if article_title else f"Artículo {article_num}",
                        'content': article_content.strip(),
                        'raw_text': article
                    })
        
        # Ordenar los artículos por relevancia descendente
        processed_articles.sort(key=lambda x: x['relevance'], reverse=True)
        
        return header, processed_articles
    
    def process_cluster(self, input_file, output_dir, model_type='bart'):
        """
        Procesa un archivo de cluster, generando resúmenes de cada artículo y un resumen final.
        
        Args:
            input_file: Ruta al archivo de cluster
            output_dir: Directorio donde guardar los resultados
            model_type: Tipo de modelo a utilizar
            
        Returns:
            Ruta al archivo de resumen final
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Leer el archivo de cluster
        with open(input_file, 'r', encoding='utf-8') as f:
            cluster_text = f.read()
        
        # Dividir en artículos
        header, articles = self.split_into_articles(cluster_text)
        
        print(f"Se han encontrado {len(articles)} artículos en el cluster")
        
        # Crear un archivo de debug para verificar la identificación de artículos
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, "articulos_detectados.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"ARTÍCULOS DETECTADOS EN: {os.path.basename(input_file)}\n\n")
            f.write(f"Encabezado:\n{header}\n\n")
            f.write(f"Total de artículos: {len(articles)}\n\n")
            
            for i, article in enumerate(articles):
                f.write(f"=== ARTÍCULO #{i+1} ===\n")
                f.write(f"Número: {article['number']}\n")
                f.write(f"Relevancia: {article['relevance']:.2f}%\n")
                f.write(f"Título: {article['title']}\n")
                f.write(f"Contenido:\n{article['content']}\n\n")
        
        # Si no se encontraron artículos, finalizar
        if not articles:
            print("No se detectaron artículos en el archivo. Revisa el formato o el separador utilizado.")
            # Crear un archivo de resumen final vacío
            final_file = os.path.join(output_dir, f"{os.path.basename(input_file).split('.')[0]}_resumen_final.txt")
            with open(final_file, 'w', encoding='utf-8') as f:
                f.write(f"No se detectaron artículos en el archivo: {input_file}\n")
                f.write("Por favor, revisa el formato del archivo de entrada.")
            return final_file
        
        # Generar resúmenes individuales de cada artículo
        article_summaries = []
        temp_dir = os.path.join(output_dir, "articulos_resumidos")
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, article in enumerate(tqdm(articles, desc="Resumiendo artículos")):
            # Solo procesar artículos con relevancia > 25% (opcional, ajustar según necesidades)
            if article['relevance'] < 25:
                continue
            
            # Generar resumen del artículo
            article_text = f"{article['title']}\n\n{article['content']}"
            summary = self.generate_summary(article_text, model_type)
            
            # Guardar resumen individual
            article_file = os.path.join(temp_dir, f"articulo_{article['number']}_resumen.txt")
            with open(article_file, 'w', encoding='utf-8') as f:
                f.write(f"Artículo {article['number']} (Relevancia: {article['relevance']:.1f}%)\n")
                f.write(f"Título: {article['title']}\n\n")
                f.write(f"RESUMEN:\n{summary}\n")
            
            # Añadir a la lista de resúmenes
            article_summaries.append({
                'number': article['number'],
                'relevance': article['relevance'],
                'title': article['title'],
                'summary': summary
            })
        
        # Crear un archivo combinado con todos los resúmenes
        combined_file = os.path.join(output_dir, "resumen_articulos_combinados.txt")
        with open(combined_file, 'w', encoding='utf-8') as f:
            # Incluir encabezado del cluster
            cluster_name = os.path.basename(input_file).split('.')[0]
            f.write(f"RESÚMENES DE ARTÍCULOS DEL CLUSTER: {cluster_name}\n\n")
            f.write(f"{header}\n\n{'='*50}\n\n")
            
            # Incluir resúmenes de artículos ordenados por relevancia
            for article in article_summaries:
                f.write(f"Artículo {article['number']} (Relevancia: {article['relevance']:.1f}%)\n")
                f.write(f"Título: {article['title']}\n")
                f.write(f"RESUMEN: {article['summary']}\n\n")
                f.write(f"{'-'*50}\n\n")
        
        # Generar resumen final de todos los resúmenes combinados
        print("Generando resumen final del cluster...")
        
        # Crear un texto que combine los resúmenes más relevantes
        final_summary_text = f"CLUSTER: {cluster_name}\n\n"
        final_summary_text += f"{header}\n\n"
        
        # Incluir los resúmenes de los artículos más relevantes (>100%)
        high_relevance_articles = [a for a in article_summaries if a['relevance'] > 100]
        if high_relevance_articles:
            final_summary_text += "RESÚMENES DE LOS ARTÍCULOS MÁS RELEVANTES:\n\n"
            for article in high_relevance_articles:
                final_summary_text += f"Artículo {article['number']} - {article['title']}\n"
                final_summary_text += f"{article['summary']}\n\n"
        else:
            # Si no hay artículos muy relevantes, incluir los top 5
            top_articles = article_summaries[:min(5, len(article_summaries))]
            final_summary_text += "RESÚMENES DE LOS ARTÍCULOS MÁS RELEVANTES:\n\n"
            for article in top_articles:
                final_summary_text += f"Artículo {article['number']} - {article['title']}\n"
                final_summary_text += f"{article['summary']}\n\n"
        
        # Generar el resumen final
        final_summary = self.generate_summary(final_summary_text, model_type)
        
        # Guardar el resumen final
        final_file = os.path.join(output_dir, f"{cluster_name}_resumen_final_{model_type}.txt")
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(f"RESUMEN FINAL DEL CLUSTER: {cluster_name}\n\n")
            f.write(f"Este resumen condensa la información de {len(article_summaries)} artículos, ")
            f.write(f"priorizando los de mayor relevancia (>{100}%).\n\n")
            f.write(f"{final_summary}\n\n")
            f.write("="*50 + "\n\n")
            f.write("ARTÍCULOS MÁS RELEVANTES:\n\n")
            
            # Listar los artículos más relevantes para referencia
            for article in high_relevance_articles if high_relevance_articles else article_summaries[:5]:
                f.write(f"- Artículo {article['number']}: {article['title']} (Relevancia: {article['relevance']:.1f}%)\n")
        
        print(f"Proceso completado. Resumen final guardado en: {final_file}")
        return final_file

def main():
    parser = argparse.ArgumentParser(description='Resumidor de clusters de artículos')
    parser.add_argument('--input', type=str, required=True, help='Archivo de cluster a resumir')
    parser.add_argument('--output', type=str, default='resumenes', help='Directorio para guardar los resúmenes')
    parser.add_argument('--model', type=str, default='bart', choices=['bart', 't5', 'pegasus', 'longformer'],
                       help='Modelo a utilizar para los resúmenes: bart (rápido), t5 (compacto), pegasus (calidad), longformer (textos largos)')
    
    args = parser.parse_args()
    
    # Crear el resumidor y procesar el cluster
    summarizer = ClusterSummarizer()
    summarizer.process_cluster(args.input, args.output, args.model)

if __name__ == "__main__":
    main()