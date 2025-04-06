#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
from transformers import pipeline

def compare_summaries(original_file, summary_files, output_file="evaluacion.txt", evaluation_model="gpt2"):
    """
    Compara resúmenes usando un modelo LLM pequeño.
    
    Args:
        original_file: Ruta al archivo de texto original
        summary_files: Lista de rutas a los archivos de resumen
        output_file: Ruta donde guardar la evaluación
    """
    # Determinar dispositivo disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizando dispositivo: {device}")
    
    # Cargar modelo pequeño de generación de texto
    try:
        print("Cargando modelo pequeño...")
        model = pipeline(
            "text-generation",
            model=evaluation_model,
            max_length=100,
            device=0 if device == 'cuda' else -1
        )
        use_model = True
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        print("Usando evaluación heurística en su lugar")
        use_model = False
    
    # Leer el archivo original
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        print(f"Archivo original leído: {len(original_text)} caracteres")
    except Exception as e:
        print(f"Error al leer archivo original: {e}")
        return

    # Leer los archivos de resumen
    summaries = {}
    for file_path in summary_files:
        try:
            # Determinar modelo basado en nombre de archivo
            file_name = os.path.basename(file_path)
            if "longformer" in file_name.lower():
                model_name = "longformer"
            elif "t5" in file_name.lower():
                model_name = "t5"
            elif "bart" in file_name.lower():
                model_name = "bart"
            else:
                model_name = os.path.splitext(file_name)[0]
                
            # Leer contenido
            with open(file_path, 'r', encoding='utf-8') as f:
                summaries[model_name] = f.read()
            print(f"Resumen leído - {model_name}: {len(summaries[model_name])} caracteres")
        except Exception as e:
            print(f"Error al leer {file_path}: {e}")
    
    # Evaluar cada resumen
    results = {}
    if use_model:
        # Evaluación con modelo LLM
        for model_name, summary in summaries.items():
            # Truncar para ajustar al modelo
            summary_preview = summary[:500] + "..." if len(summary) > 500 else summary
            original_preview = original_text[:500] + "..." if len(original_text) > 500 else original_text
            
            # Crear prompt para la evaluación
            prompt = (
                f"Evalúa la calidad de este resumen del 1 al 10:\n\n"
                f"TEXTO ORIGINAL (extracto):\n{original_preview}\n\n"
                f"RESUMEN ({model_name}):\n{summary_preview}\n\n"
                f"PUNTUACIÓN (1-10): "
            )
            
            # Generar evaluación
            try:
                output = model(prompt, max_new_tokens=20, num_return_sequences=1)
                generated_text = output[0]['generated_text'][len(prompt):].strip()
                
                # Extraer puntuación numérica
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', generated_text)
                if score_match:
                    score = float(score_match.group(1))
                    # Limitar al rango 1-10
                    score = max(1.0, min(10.0, score))
                else:
                    # Asignar valor neutro si no se encuentra puntuación
                    score = 5.0
                
                results[model_name] = {
                    "score": score,
                    "explanation": generated_text
                }
                print(f"Evaluación para {model_name}: {score}/10")
            except Exception as e:
                print(f"Error evaluando {model_name}: {e}")
                results[model_name] = {"score": 5.0, "explanation": f"Error: {str(e)}"}
    else:
        # Evaluación heurística simple (sin modelo)
        for model_name, summary in summaries.items():
            # Métricas simples
            coverage = calculate_coverage(original_text, summary)
            conciseness = calculate_conciseness(summary) 
            coherence = calculate_coherence(summary)
            
            # Puntuación ponderada
            score = (coverage * 0.5) + (conciseness * 0.3) + (coherence * 0.2)
            
            results[model_name] = {
                "score": score,
                "metrics": {
                    "coverage": coverage,
                    "conciseness": conciseness,
                    "coherence": coherence
                },
                "explanation": f"Cobertura: {coverage:.1f}/10, Concisión: {conciseness:.1f}/10, Coherencia: {coherence:.1f}/10"
            }
            print(f"Evaluación para {model_name}: {score:.1f}/10")
    
    # Determinar el mejor resumen
    if results:
        best_model = max(results.items(), key=lambda x: x[1]["score"])[0]
        print(f"\nMejor resumen: {best_model} con puntuación {results[best_model]['score']:.1f}/10")
    else:
        best_model = None
        print("No se pudo evaluar ningún resumen")
    
    # Guardar resultados
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("EVALUACIÓN DE RESÚMENES\n\n")
        
        if best_model:
            f.write(f"EL MEJOR RESUMEN ES: {best_model} - " +
                   f"Puntuación: {results[best_model]['score']:.1f}/10\n\n")
            
        f.write("RESULTADOS DETALLADOS:\n\n")
        for model_name, result in results.items():
            f.write(f"{model_name}: {result['score']:.1f}/10\n")
            if "explanation" in result:
                f.write(f"- {result['explanation']}\n")
            if "metrics" in result:
                metrics = result["metrics"]
                f.write(f"- Cobertura: {metrics['coverage']:.1f}/10\n")
                f.write(f"- Concisión: {metrics['conciseness']:.1f}/10\n")
                f.write(f"- Coherencia: {metrics['coherence']:.1f}/10\n")
            f.write("\n")
    
    print(f"Resultados guardados en {output_file}")
    return best_model, results

# Funciones de evaluación heurística
def calculate_coverage(original_text, summary):
    """Calcula cobertura del contenido original"""
    original_words = set(w.lower() for w in original_text.split() if len(w) > 4)
    summary_words = set(w.lower() for w in summary.split() if len(w) > 4)
    
    if not original_words:
        return 5.0
    
    coverage = len(summary_words.intersection(original_words)) / min(len(original_words), 100)
    return min(10.0, coverage * 20)

def calculate_conciseness(summary):
    """Calcula concisión del resumen"""
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

def calculate_coherence(summary):
    """Estima coherencia del resumen"""
    sentences = summary.split('.')
    
    # Calcular longitud promedio de oraciones
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    
    # Puntuación de coherencia basada en longitud de oraciones
    if avg_sentence_length < 5:
        coherence = 4.0  # Oraciones muy cortas
    elif avg_sentence_length < 10:
        coherence = 6.0  # Oraciones cortas
    elif avg_sentence_length < 20:
        coherence = 8.0  # Longitud ideal
    else:
        coherence = 5.0  # Oraciones muy largas
    
    # Penalizar si hay muy pocas oraciones
    if len(sentences) < 3:
        coherence *= 0.7
        
    return min(10.0, coherence)

if __name__ == "__main__":
    # Procesar argumentos
    if len(sys.argv) < 3:
        print("Uso: python simple_evaluator.py <archivo_original> <resumen1> [resumen2] ... [--output <archivo_salida>]")
        sys.exit(1)
    
    # Obtener argumentos
    original_file = sys.argv[1]
    summary_files = []
    output_file = "evaluacion_resumenes.txt"
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i+1 < len(sys.argv):
            output_file = sys.argv[i+1]
            i += 2
        else:
            summary_files.append(sys.argv[i])
            i += 1
    
    # Ejecutar evaluación
    compare_summaries(original_file, summary_files, output_file)