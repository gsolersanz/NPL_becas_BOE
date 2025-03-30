#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class ResumenevaluatorLLM:
    def __init__(self, model_name="distilgpt2", device=None):
        """
        Inicializa el evaluador de resúmenes basado en un modelo LLM ligero.
        
        Args:
            model_name: Nombre o tipo del modelo LLM a utilizar
            device: Dispositivo a utilizar (None para detección automática)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inicializando evaluador LLM con modelo {model_name} en dispositivo {self.device}")
        
        # Indicar si debemos usar el método heurístico
        self.use_heuristic = False
        
        # Cargar modelo pequeño por defecto
        try:
            if model_name == "tiny":
                # Modelo extremadamente pequeño para computadoras con memoria muy limitada
                self.pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",  # ~550MB
                    max_length=512,
                    device=0 if self.device == 'cuda' else -1
                )
                print("Cargado modelo DistilGPT2 (pequeño, ~550MB)")
            elif model_name == "small":
                # Modelo pequeño para computadoras con memoria limitada
                self.pipeline = pipeline(
                    "text-generation",
                    model="gpt2",       # ~550MB
                    max_length=512,
                    device=0 if self.device == 'cuda' else -1
                )
                print("Cargado modelo GPT2 (pequeño, ~550MB)")
            elif model_name == "distilbert":
                # Alternativa basada en BERT (podría ser mejor para español)
                self.pipeline = pipeline(
                    "text-classification",
                    model="dccuchile/bert-base-spanish-wwm-uncased",
                    device=0 if self.device == 'cuda' else -1
                )
                # Para este modelo, usaremos un enfoque diferente de evaluación
                self.using_classification = True
                print("Cargado modelo BERT español (clasificación)")
            else:
                # Intentar cargar el modelo especificado
                print(f"Intentando cargar modelo personalizado: {model_name}")
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    max_length=512,
                    device=0 if self.device == 'cuda' else -1
                )
                self.using_classification = False
                
            print(f"Modelo LLM cargado correctamente: {model_name}")
            
        except Exception as e:
            print(f"Error al cargar el modelo {model_name}: {e}")
            print("La evaluación se realizará usando heurísticas sin modelo (más liviano)")
            self.use_heuristic = True
    
    def _simple_evaluation_method(self, original_text, summaries):
        """
        Método alternativo de evaluación para cuando no se pueden usar LLMs grandes.
        Evalúa resúmenes usando heurísticas simples y modelos más pequeños.
        """
        scores = {}
        reasoning = ""
        
        # Calcular puntuación para cada resumen
        for model_name, summary in summaries.items():
            # Medidas simples
            coverage_score = self._calculate_coverage(original_text, summary)
            conciseness_score = self._calculate_conciseness(summary)
            coherence_score = self._calculate_coherence(summary)
            
            # Puntuación total (ponderada)
            total_score = (0.5 * coverage_score + 
                          0.3 * conciseness_score + 
                          0.2 * coherence_score)
            
            scores[model_name] = total_score
            
            # Añadir explicación a razonamiento
            reasoning += f"Resumen {model_name}: {total_score:.2f}/10\n"
            reasoning += f"- Cobertura: {coverage_score:.2f}/10\n"
            reasoning += f"- Concisión: {conciseness_score:.2f}/10\n"
            reasoning += f"- Coherencia: {coherence_score:.2f}/10\n\n"
        
        # Encontrar el mejor modelo
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        best_score = scores[best_model]
        
        # Añadir conclusión
        reasoning += f"CONCLUSIÓN:\nEl mejor resumen es el generado por el modelo {best_model} "
        reasoning += f"con una puntuación de {best_score:.2f}/10 porque tiene "
        reasoning += "la mejor combinación de cobertura de contenido, concisión y coherencia."
        
        return {
            "mejor_modelo": best_model,
            "puntuaciones": scores,
            "razonamiento": reasoning,
            "evaluacion_completa": reasoning
        }
    
    def _calculate_coverage(self, original_text, summary):
        """
        Calcula qué tan bien el resumen cubre el contenido importante del texto original.
        """
        # Extraer palabras clave del texto original (simplificado)
        original_words = set(w.lower() for w in original_text.split() if len(w) > 4)
        summary_words = set(w.lower() for w in summary.split() if len(w) > 4)
        
        # Calcular coincidencia de palabras clave
        if len(original_words) == 0:
            return 5.0  # Puntuación media si no hay palabras
        
        coverage = len(summary_words.intersection(original_words)) / min(len(original_words), 100)
        
        # Convertir a escala de 10 puntos
        return min(10.0, coverage * 20)
    
    def _calculate_conciseness(self, summary):
        """
        Calcula la concisión del resumen.
        """
        # Un buen resumen debe ser conciso (< 20% del texto original)
        words = len(summary.split())
        
        if words < 50:
            return 9.0  # Muy conciso
        elif words < 150:
            return 8.0  # Conciso
        elif words < 300:
            return 6.0  # Aceptable
        elif words < 500:
            return 4.0  # Algo extenso
        else:
            return 2.0  # Demasiado largo
    
    def _calculate_coherence(self, summary):
        """
        Estima la coherencia del resumen basándose en heurísticas simples.
        """
        sentences = summary.split('.')
        
        # Verificar longitud promedio de oraciones
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Puntuación base de coherencia
        if avg_sentence_length < 5:
            coherence_score = 4.0  # Oraciones muy cortas, probablemente fragmentadas
        elif avg_sentence_length < 10:
            coherence_score = 6.0  # Oraciones cortas
        elif avg_sentence_length < 20:
            coherence_score = 8.0  # Longitud de oración ideal
        else:
            coherence_score = 5.0  # Oraciones demasiado largas
        
        # Penalizar resúmenes con muy pocas oraciones
        if len(sentences) < 3:
            coherence_score *= 0.7
            
        return min(10.0, coherence_score)
    
    def evaluate_summaries(self, original_text, summaries, max_tokens=800):
        """
        Evalúa múltiples resúmenes y determina cuál es el mejor.
        
        Args:
            original_text: Texto original completo
            summaries: Diccionario con los resúmenes (clave: nombre del modelo, valor: texto del resumen)
            max_tokens: Longitud máxima para truncar textos demasiado largos
            
        Returns:
            Diccionario con el mejor resumen, puntuaciones y razones
        """
        # Truncar textos si son demasiado largos
        if len(original_text) > max_tokens * 5:  # Limitar texto original
            print(f"Truncando texto original (demasiado largo)")
            # Tomar el principio y el final para cubrir ambas partes
            original_extract = original_text[:max_tokens * 2] + "\n...\n" + original_text[-max_tokens * 2:]
        else:
            original_extract = original_text
        
        truncated_summaries = {}
        for model, summary in summaries.items():
            if len(summary) > max_tokens:
                truncated_summaries[model] = summary[:max_tokens] + "..."
            else:
                truncated_summaries[model] = summary
        
        # Usar método heurístico si está configurado
        if self.use_heuristic:
            return self._simple_evaluation_method(original_text, truncated_summaries)
        
        # Generar el prompt para el LLM pequeño
        prompt = self._generate_compact_evaluation_prompt(original_extract, truncated_summaries)
        
        # Generar evaluación con el LLM ligero
        try:
            result = self._generate_lightweight_evaluation(prompt)
            evaluation = self._parse_evaluation_result(result, list(summaries.keys()))
            return evaluation
        except Exception as e:
            print(f"Error en la generación con LLM: {e}")
            print("Usando método de respaldo basado en heurísticas...")
            return self._simple_evaluation_method(original_text, truncated_summaries)
    
    def _generate_compact_evaluation_prompt(self, original_text, summaries):
        """
        Genera un prompt compacto para evaluación con modelos pequeños.
        """
        # Crear un prompt más corto y directo para modelos pequeños
        prompt = (
            "Evalúa estos resúmenes y determina cuál es el mejor:\n\n"
            "TEXTO ORIGINAL (extracto):\n"
            f"{original_text[:800]}...\n\n"
            "RESÚMENES:\n"
        )
        
        for model, summary in summaries.items():
            prompt += f"[{model.upper()}]:\n{summary[:400]}...\n\n"
        
        prompt += (
            "EVALUACIÓN:\n"
            "El mejor resumen es el del modelo: "
        )
        
        return prompt
    
    def _generate_lightweight_evaluation(self, prompt):
        """
        Genera evaluación con un modelo ligero.
        """
        if self.use_heuristic:
            raise ValueError("No hay modelo disponible para evaluación")
            
        # Generar texto con configuración de generación ligera
        try:
            outputs = self.pipeline(
                prompt, 
                max_length=len(prompt) + 200,
                num_return_sequences=1
            )
            
            # Extraer resultado (el texto generado sin el prompt)
            result = outputs[0]['generated_text'][len(prompt):]
            return result
        except Exception as e:
            print(f"Error en la generación: {e}")
            return "Error en la evaluación"
    
    def _parse_evaluation_result(self, result, model_names):
        """
        Analiza el resultado del modelo pequeño para extraer el mejor modelo.
        """
        best_model = None
        
        # Buscar menciones de modelos
        for model in model_names:
            if model.lower() in result.lower():
                best_model = model
                break
                
        # Si no se encontró un modelo, intentar con otro enfoque
        if not best_model and model_names:
            # Primera opción: buscar el nombre del modelo en mayúsculas
            for model in model_names:
                if model.upper() in result:
                    best_model = model
                    break
                    
            # Segunda opción: elegir el primer modelo
            if not best_model:
                best_model = model_names[0]
        
        return {
            "mejor_modelo": best_model,
            "puntuaciones": {},  # No tenemos puntuaciones específicas
            "razonamiento": result,
            "evaluacion_completa": f"Evaluación del modelo:\n{result}"
        }

    def evaluate_summary_files(self, original_file, summary_folder, output_file=None):
        """
        Evalúa los resúmenes guardados en archivos y determina el mejor.
        
        Args:
            original_file: Ruta al archivo de texto original
            summary_folder: Carpeta que contiene los archivos de resúmenes
            output_file: Archivo donde guardar los resultados (opcional)
            
        Returns:
            Diccionario con los resultados de la evaluación
        """
        # Leer el archivo original
        with open(original_file, 'r', encoding='utf-8') as file:
            original_text = file.read()
        
        # Obtener nombre base del archivo original
        base_name = Path(original_file).stem
        
        # Buscar archivos de resumen relacionados
        summary_files = list(Path(summary_folder).glob(f"{base_name}*_summary.txt"))
        
        if not summary_files:
            print(f"No se encontraron resúmenes para {base_name} en {summary_folder}")
            return None
        
        # Leer resúmenes
        summaries = {}
        for summary_file in summary_files:
            # Extraer nombre del modelo del nombre del archivo
            # Formato esperado: nombre_modelo_summary.txt
            file_name = summary_file.stem
            model_name = file_name.split('_')[-2] if '_' in file_name else 'unknown'
            
            # Leer resumen
            with open(summary_file, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Extraer solo la parte del resumen (después de "Resumen:")
                if "Resumen:" in content:
                    summary = content.split("Resumen:")[1].strip()
                else:
                    summary = content
                
                summaries[model_name] = summary
        
        # Evaluar resúmenes
        evaluation = self.evaluate_summaries(original_text, summaries)
        
        # Guardar resultados
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(f"EVALUACIÓN DE RESÚMENES PARA: {base_name}\n\n")
                file.write(f"MEJOR RESUMEN: {evaluation['mejor_modelo']}\n\n")
                file.write(f"RAZONAMIENTO:\n{evaluation['razonamiento']}\n\n")
                
                # Incluir el mejor resumen al final
                if evaluation['mejor_modelo'] in summaries:
                    file.write(f"TEXTO DEL MEJOR RESUMEN ({evaluation['mejor_modelo']}):\n")
                    file.write(summaries[evaluation['mejor_modelo']])
        
        return {
            "archivo_original": str(original_file),  # Convertir Path a string para evitar problemas con JSON
            "nombre_base": base_name,
            "mejor_modelo": evaluation['mejor_modelo'],
            "mejor_resumen": summaries.get(evaluation['mejor_modelo'], ""),
            "evaluacion": evaluation
        }

if __name__ == "__main__":
    import argparse
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Evaluación de resúmenes con modelos ligeros')
    parser.add_argument('--original', type=str, required=True, help='Archivo original o directorio con archivos originales')
    parser.add_argument('--summaries', type=str, required=True, help='Directorio con archivos de resúmenes')
    parser.add_argument('--output', type=str, default='evaluacion_resumenes', help='Directorio para guardar evaluaciones')
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small', 'distilbert', 'distilgpt2', 'gpt2'], 
                       help='Modelo de evaluación a utilizar')
    parser.add_argument('--force-heuristic', action='store_true', help='Forzar uso de heurísticas en lugar de modelo')
    
    args = parser.parse_args()
    
    # Crear evaluador
    evaluator = ResumenevaluatorLLM(model_name=args.model)
    
    # Forzar el modo heurístico si se solicita
    if args.force_heuristic:
        evaluator.use_heuristic = True
        print("Utilizando evaluación basada en heurísticas por solicitud del usuario")
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    # Procesar archivo o directorio
    if os.path.isfile(args.original):
        # Evaluar un solo archivo
        output_file = os.path.join(args.output, f"{Path(args.original).stem}_evaluacion.txt")
        result = evaluator.evaluate_summary_files(args.original, args.summaries, output_file)
        
        if result:
            print(f"\nMejor resumen para {result['nombre_base']}: {result['mejor_modelo']}")
            print(f"Evaluación guardada en {output_file}")
    
    elif os.path.isdir(args.original):
        # Evaluar todos los archivos en el directorio
        original_files = list(Path(args.original).glob('*.txt'))
        
        results = []
        for original_file in tqdm(original_files, desc="Evaluando archivos"):
            output_file = os.path.join(args.output, f"{original_file.stem}_evaluacion.txt")
            result = evaluator.evaluate_summary_files(original_file, args.summaries, output_file)
            
            if result:
                results.append(result)
                print(f"\nMejor resumen para {result['nombre_base']}: {result['mejor_modelo']}")
        
        # Guardar resumen general
        summary_file = os.path.join(args.output, "evaluacion_general.json")
        with open(summary_file, 'w', encoding='utf-8') as file:
            # Asegurar que solo hay datos serializables a JSON
            json.dump(results, file, ensure_ascii=False, indent=2)
        
        print(f"\nEvaluación general guardada en {summary_file}")
    
    else:
        print(f"Error: La ruta {args.original} no existe o no es válida")