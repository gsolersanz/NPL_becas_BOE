#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path
import torch
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    LEDForConditionalGeneration, LEDTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
from tqdm import tqdm
# Añadir en la parte superior del script después de las importaciones
import huggingface_hub
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 30  # Aumentar a 30 segundos

class SummarizationModels:
    def __init__(self, device=None):
        """
        Inicializa los modelos de resumen.
        
        Args:
            device: Dispositivo a utilizar para los modelos (cuda o cpu)
        """
        # Determinar dispositivo (GPU o CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilizando dispositivo: {self.device}")
        
        # Inicializar modelos como None (se cargarán según necesidad)
        self.models = {
            'bart': {
                'name': 'facebook/bart-large-cnn',
                'model': None,
                'tokenizer': None,
                'max_length': 1024,
                'min_length': 50,
                'length_penalty': 2.0,
                'num_beams': 4
            },
            't5': {
                'name': 't5-base',
                'model': None,
                'tokenizer': None,
                'max_length': 512,
                'min_length': 50,
                'length_penalty': 2.0,
                'num_beams': 4
            },
            'pegasus': {
                'name': 'google/pegasus-xsum',
                'model': None,
                'tokenizer': None,
                'max_length': 512,
                'min_length': 50,
                'length_penalty': 1.0,
                'num_beams': 4
            },
            'led': {
                'name': 'allenai/led-base-16384',
                'model': None,
                'tokenizer': None,
                'max_length': 1024,
                'min_length': 50,
                'length_penalty': 2.0,
                'num_beams': 4,
                'global_attention_indices': [0]  # Atención global para el primer token
            },
            'prophetnet': {
                'name': 'microsoft/prophetnet-large-uncased-cnndm',
                'model': None,
                'tokenizer': None,
                'max_length': 512,
                'min_length': 50,
                'length_penalty': 2.0,
                'num_beams': 4
            }
        }
    
    def load_model(self, model_type):
        """
        Carga un modelo específico bajo demanda.
        
        Args:
            model_type: Tipo de modelo ('bart', 't5', 'pegasus', 'led', 'prophetnet')
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
            
            elif model_type == 'led':
                config['tokenizer'] = LEDTokenizer.from_pretrained(config['name'])
                config['model'] = LEDForConditionalGeneration.from_pretrained(config['name']).to(self.device)
            
            elif model_type == 'prophetnet':
                config['tokenizer'] = AutoTokenizer.from_pretrained(config['name'])
                config['model'] = AutoModelForSeq2SeqLM.from_pretrained(config['name']).to(self.device)
            
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
        
        # Tokenizar el texto
        inputs = tokenizer(text_prepared, max_length=config['max_length'], 
                           truncation=True, padding="longest", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Configuración específica para LED (atención global)
        if model_type == 'led':
            global_attention_mask = torch.zeros_like(inputs['input_ids'])
            # Establecer atención global en posiciones específicas (primer token, tokens de puntuación, etc.)
            for idx in config['global_attention_indices']:
                global_attention_mask[:, idx] = 1
            inputs['global_attention_mask'] = global_attention_mask
        
        # Generar resumen
        # En la función generate_summary:
        summary_ids = model.generate(
            **inputs,
            max_length=min(config['max_length'], 512),  # Limitar para documentos legales
            min_length=150,  # Aumentar para obtener resúmenes más completos
            length_penalty=1.5,  # Ajustar para equilibrar longitud y calidad
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3  # Evitar repeticiones
        )
        
        # Decodificar el resumen
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def summarize_file(self, file_path, output_dir=None):
        """
        Genera resúmenes para un archivo usando todos los modelos.
        
        Args:
            file_path: Ruta al archivo a resumir
            output_dir: Directorio donde guardar los resúmenes (opcional)
            
        Returns:
            Diccionario con los resúmenes generados por cada modelo
        """
        # Leer el archivo
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        file_name = Path(file_path).stem
        print(f"Generando resúmenes para {file_name}...")
        
        # Generar resúmenes con cada modelo
        summaries = {}
        for model_type in tqdm(self.models.keys(), desc="Modelos"):
            try:
                summary = self.generate_summary(text, model_type)
                summaries[model_type] = summary
                
                # Guardar resumen si se especificó un directorio de salida
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{file_name}_{model_type}_summary.txt")
                    
                    with open(output_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(f"Archivo original: {file_path}\n")
                        out_file.write(f"Modelo: {model_type} ({self.models[model_type]['name']})\n")
                        out_file.write(f"Resumen:\n\n{summary}")
                    
                    print(f"Resumen {model_type} guardado en {output_path}")
            
            except Exception as e:
                print(f"Error al generar resumen con {model_type}: {e}")
                summaries[model_type] = f"Error: {e}"
        
        return summaries
    
    def summarize_directory(self, input_dir, output_dir):
        """
        Genera resúmenes para todos los archivos de texto en un directorio.
        
        Args:
            input_dir: Directorio con los archivos a resumir
            output_dir: Directorio donde guardar los resúmenes
        """
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener archivos de texto
        files = list(Path(input_dir).glob('*.txt'))
        
        if not files:
            print(f"No se encontraron archivos de texto en {input_dir}")
            return
        
        print(f"Procesando {len(files)} archivos de texto...")
        
        # Generar resúmenes para cada archivo
        all_summaries = {}
        for file_path in tqdm(files, desc="Archivos"):
            file_name = file_path.stem
            all_summaries[file_name] = self.summarize_file(file_path, output_dir)
        
        # Crear un archivo de resumen general
        summary_path = os.path.join(output_dir, "todos_los_resumenes.txt")
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write("RESUMEN DE TODOS LOS ARCHIVOS\n\n")
            
            for file_name, models in all_summaries.items():
                summary_file.write(f"\n{'=' * 80}\n")
                summary_file.write(f"ARCHIVO: {file_name}\n")
                summary_file.write(f"{'=' * 80}\n\n")
                
                for model_type, summary in models.items():
                    summary_file.write(f"\n{'-' * 40}\n")
                    summary_file.write(f"Modelo: {model_type}\n")
                    summary_file.write(f"{'-' * 40}\n\n")
                    summary_file.write(f"{summary}\n")
        
        print(f"Resumen general guardado en {summary_path}")
        return all_summaries

# Función principal
def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Generación de resúmenes con modelos de Hugging Face')
    parser.add_argument('--input', type=str, nargs='+', required=True, help='Archivo(s) o directorio de entrada')
    parser.add_argument('--output', type=str, default='resumenes', help='Directorio de salida para los resúmenes')
    parser.add_argument('--models', type=str, nargs='+', default=['bart', 't5', 'pegasus', 'led', 'prophetnet'],
                      help='Modelos a utilizar (bart, t5, pegasus, led, prophetnet)')
    
    args = parser.parse_args()
    
    # Crear instancia de los modelos
    summarizer = SummarizationModels()
    
    # Filtrar modelos seleccionados
    summarizer.models = {k: v for k, v in summarizer.models.items() if k in args.models}
    
    # Procesar entradas
    for input_path_str in args.input:
        input_path = Path(input_path_str)
        if input_path.is_file():
            # Procesar un solo archivo
            print(f"Procesando archivo: {input_path}")
            summarizer.summarize_file(input_path, args.output)
        elif input_path.is_dir():
            # Procesar un directorio
            print(f"Procesando directorio: {input_path}")
            summarizer.summarize_directory(input_path, args.output)
        else:
            print(f"Error: La ruta de entrada {input_path} no existe")

if __name__ == "__main__":
    main()