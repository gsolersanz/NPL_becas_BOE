#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para ejecutar el pipeline completo de análisis de documentos de becas.
Este script comprueba dependencias e inicia la interfaz de usuario.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def verificar_dependencias():
    """Verifica las dependencias necesarias y las instala si faltan."""
    dependencias_requeridas = [
        "transformers",      # Para modelos de transformers y resumen
        "torch",             # Backend para transformers
        "sentence_transformers",  # Para el modelado de temas
        "scikit-learn",      # Para clustering y PCA
        "nltk",              # Procesamiento de lenguaje natural
        "pandas",            # Manipulación de datos
        "matplotlib",        # Visualización
        "seaborn",           # Gráficos mejorados
        "tqdm",              # Barras de progreso
        "customtkinter"      # Interfaz mejorada (opcional)
    ]
    
    dependencias_faltantes = []
    
    print("Verificando dependencias...")
    
    for dep in dependencias_requeridas:
        try:
            importlib.import_module(dep.replace('-', '_'))  # Manejo nombres como sentence-transformers
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} (falta)")
            dependencias_faltantes.append(dep)
    
    if dependencias_faltantes:
        print("\nSe instalarán las dependencias faltantes:")
        for dep in dependencias_faltantes:
            print(f"- {dep}")
        
        respuesta = input("\n¿Desea instalar estas dependencias ahora? (s/n): ")
        if respuesta.lower() in ['s', 'si', 'sí', 'y', 'yes']:
            for dep in dependencias_faltantes:
                print(f"\nInstalando {dep}...")
                subprocess.run([sys.executable, "-m", "pip", "install", dep])
        else:
            print("\nPor favor, instale manualmente las dependencias faltantes antes de continuar.")
            sys.exit(1)
    
    # Descargar recursos adicionales
    print("\nVerificando recursos adicionales de NLTK...")
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ Tokenizador NLTK disponible")
        except LookupError:
            print("Descargando tokenizador NLTK...")
            nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Error al verificar recursos NLTK: {e}")
    
    print("\nTodas las dependencias están instaladas correctamente.")

def verificar_archivos():
    """Verifica que existan los archivos del pipeline."""
    archivos_necesarios = [
        "transformer_topic_modeling.py",
        "summarization_models.py",
        "llm_evaluator.py",
        "interfaz_usuario.py"
    ]
    
    archivos_faltantes = []
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    print("\nVerificando archivos del pipeline...")
    
    for archivo in archivos_necesarios:
        ruta = os.path.join(directorio_actual, archivo)
        if not os.path.exists(ruta):
            print(f"✗ {archivo} (falta)")
            archivos_faltantes.append(archivo)
        else:
            print(f"✓ {archivo}")
    
    if archivos_faltantes:
        print("\nERROR: Faltan archivos necesarios para el pipeline.")
        print("Por favor, asegúrate de que todos los archivos están en el mismo directorio.")
        sys.exit(1)
    
    print("\nTodos los archivos del pipeline están presentes.")

def crear_directorios_necesarios():
    """Crea los directorios necesarios para el funcionamiento del pipeline."""
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    directorios = [
        os.path.join(directorio_actual, "corpus_txt"),
        os.path.join(directorio_actual, "resultados")
    ]
    
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
        print(f"Directorio creado/verificado: {directorio}")
    
    return directorios[0]  # Devuelve el directorio corpus_txt

def crear_archivos_demo(corpus_dir):
    """Crea archivos de demostración si no hay archivos para analizar."""
    textos_existentes = list(Path(corpus_dir).glob("*.txt"))
    
    if not textos_existentes:
        print("\nNo se encontraron archivos de texto (.txt) en el directorio corpus_txt.")
        respuesta = input("¿Desea crear archivos de demostración? (s/n): ")
        
        if respuesta.lower() in ['s', 'si', 'sí', 'y', 'yes']:
            # Crear un archivo de demostración
            texto_demo = """
CONVOCATORIA DE BECAS PARA ESTUDIANTES UNIVERSITARIOS

REQUISITOS ACADÉMICOS:
- Estar matriculado en un mínimo de 60 créditos.
- Haber obtenido una nota media mínima de 5,00 puntos en el curso anterior.
- Para estudios de grado de la rama de Ingeniería/Arquitectura, se deberá acreditar haber superado el 65% de los créditos matriculados.
- Para estudios de grado de la rama de Ciencias, se deberá acreditar haber superado el 65% de los créditos matriculados.
- Para estudios de grado de la rama de Ciencias de la Salud, se deberá acreditar haber superado el 80% de los créditos matriculados.
- Para estudios de grado de la rama de Ciencias Sociales y Jurídicas o Artes y Humanidades, se deberá acreditar haber superado el 90% de los créditos matriculados.

REQUISITOS ECONÓMICOS:
- No superar los umbrales de renta familiar establecidos:
  * Familias de un miembro: 14.112,00 euros
  * Familias de dos miembros: 24.089,00 euros
  * Familias de tres miembros: 32.697,00 euros
  * Familias de cuatro miembros: 38.831,00 euros
  * Familias de cinco miembros: 43.402,00 euros
  * Familias de seis miembros: 46.853,00 euros
  * Familias de siete miembros: 50.267,00 euros
  * Familias de ocho miembros: 53.665,00 euros
- No superar los umbrales de patrimonio familiar:
  * La suma de los valores catastrales de las fincas urbanas que pertenezcan a la unidad familiar, excluida la vivienda habitual, no podrá superar los 42.900,00 euros.
  * La suma de los valores catastrales de las fincas rústicas que pertenezcan a la unidad familiar no podrá superar los 13.130,00 euros por cada miembro computable.

DOCUMENTACIÓN A PRESENTAR:
- Solicitud debidamente cumplimentada.
- Fotocopia del DNI/NIE de todos los miembros de la unidad familiar.
- Documentación acreditativa de la situación académica.
- Documentación acreditativa de los ingresos económicos (declaración de la renta o certificado de imputaciones).
- En su caso, título de familia numerosa, certificado de discapacidad, etc.

PLAZOS:
- El plazo de presentación de solicitudes será del 1 de agosto al 14 de octubre de 2021.
- Las solicitudes deberán presentarse a través de la sede electrónica del Ministerio de Educación y Formación Profesional.
- No se admitirán solicitudes fuera de plazo, salvo casos excepcionales debidamente justificados.

PROCESO DE ADJUDICACIÓN:
- Las solicitudes serán evaluadas por las comisiones de selección de becarios.
- Las resoluciones de concesión o denegación se notificarán a través de la sede electrónica.
- Contra la resolución de concesión o denegación, se podrá interponer recurso de reposición en el plazo de un mes.
"""
            
            demo_file_path = os.path.join(corpus_dir, "ejemplo_becas.txt")
            with open(demo_file_path, 'w', encoding='utf-8') as f:
                f.write(texto_demo)
            
            print(f"\nSe ha creado un archivo de demostración: {demo_file_path}")
            print("Puedes usarlo para probar el pipeline.")
            
            return True
    
    return len(textos_existentes) > 0  # Devuelve True si hay archivos existentes

def ejecutar_modo_consola(corpus_dir, resultados_dir):
    """Ejecuta el pipeline en modo consola."""
    # Solicitar información básica
    print("\n=== MODO CONSOLA DEL PIPELINE ===")
    
    # Listar archivos disponibles
    archivos = list(Path(corpus_dir).glob("*.txt"))
    if not archivos:
        print("No hay archivos de texto disponibles en el directorio.")
        return
    
    print("\nArchivos disponibles:")
    for i, archivo in enumerate(archivos):
        print(f"{i+1}. {archivo.name}")
    
    seleccion = input(f"\nSeleccione un archivo (1-{len(archivos)}): ")
    try:
        idx = int(seleccion) - 1
        if idx < 0 or idx >= len(archivos):
            raise ValueError()
        archivo_seleccionado = str(archivos[idx])
    except ValueError:
        print("Selección inválida. Usando el primer archivo.")
        archivo_seleccionado = str(archivos[0])
    
    num_topics = input("Número de temas a identificar [5]: ")
    num_topics = int(num_topics) if num_topics.strip() and num_topics.strip().isdigit() else 5
    
    llm_model = input("Modelo LLM para evaluación (gpt2, distilgpt2, small) [gpt2]: ")
    llm_model = llm_model.strip() if llm_model.strip() in ['gpt2', 'distilgpt2', 'small'] else 'gpt2'
    
    # Ejecutar pipeline paso a paso
    print("\n=== INICIANDO PROCESAMIENTO ===")
    
    # Paso 1: Análisis de temas
    print("\nPaso 1: Realizando análisis de temas...")
    temas_dir = os.path.join(resultados_dir, "1_temas")
    os.makedirs(temas_dir, exist_ok=True)
    
    try:
        # Importar directamente el módulo
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from transformer_topic_modeling import BecasTransformerTopicModel
        
        # Crear y ejecutar el modelador de temas
        modeler = BecasTransformerTopicModel()
        
        # Cargar documento
        with open(archivo_seleccionado, 'r', encoding='utf-8') as file:
            texto = file.read()
            modeler.raw_texts.append(texto)
            modeler.doc_names.append(Path(archivo_seleccionado).stem)
        
        # Analizar documento
        results = modeler.analyze_documents(n_clusters=num_topics)
        
        # Guardar resultados
        modeler.save_topic_sections(temas_dir)
        modeler.visualize_topics(os.path.join(resultados_dir, "distribucion_temas.png"))
        
        # Paso 2: Generar resúmenes
        print("\nPaso 2: Generando resúmenes...")
        resumenes_dir = os.path.join(resultados_dir, "2_resumenes")
        os.makedirs(resumenes_dir, exist_ok=True)
        
        from summarization_models import SummarizationModels
        
        # Inicializar el resumidor
        summarizer = SummarizationModels()
        
        # Procesar archivos de temas
        archivos_tema = list(Path(temas_dir).glob("*.txt"))
        for archivo_tema in archivos_tema:
            print(f"Generando resúmenes para {archivo_tema.name}...")
            summarizer.summarize_file(str(archivo_tema), resumenes_dir)
        
        # Paso 3: Evaluar resúmenes
        print("\nPaso 3: Evaluando resúmenes...")
        eval_dir = os.path.join(resultados_dir, "3_evaluaciones")
        os.makedirs(eval_dir, exist_ok=True)
        
        from llm_evaluator import ResumenevaluatorLLM
        
        # Inicializar evaluador
        evaluator = ResumenevaluatorLLM(model_name=llm_model)
        
        # Evaluar resúmenes para cada archivo de tema
        for archivo_tema in archivos_tema:
            print(f"Evaluando resúmenes para {archivo_tema.name}...")
            
            # Buscar archivos de resumen relacionados con este tema
            patron_busqueda = f"{archivo_tema.stem}*_summary.txt"
            archivos_resumen = list(Path(resumenes_dir).glob(patron_busqueda))
            
            if archivos_resumen:
                # Ruta para guardar la evaluación
                output_file = os.path.join(eval_dir, f"{archivo_tema.stem}_evaluacion.txt")
                
                # Evaluar
                evaluator.evaluate_summary_files(str(archivo_tema), resumenes_dir, output_file)
            else:
                print(f"No se encontraron resúmenes para {archivo_tema.name}")
        
        print(f"\nProceso completado. Resultados guardados en: {resultados_dir}")
        
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Función principal."""
    print("\n" + "="*80)
    print("ANALIZADOR DE DOCUMENTOS DE BECAS - PIPELINE COMPLETO")
    print("="*80 + "\n")
    
    # Verificar dependencias
    verificar_dependencias()
    
    # Verificar archivos del pipeline
    verificar_archivos()
    
    # Crear directorios necesarios
    corpus_dir = crear_directorios_necesarios()
    resultados_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    
    # Crear archivos de demostración si es necesario
    hay_archivos = crear_archivos_demo(corpus_dir)
    
    if not hay_archivos:
        print("\nNecesitas tener al menos un archivo de texto para analizar.")
        sys.exit(1)
    
    # Iniciar interfaz de usuario
    print("\nIniciando interfaz de usuario...")
    
    try:
        # Importar y ejecutar interfaz_usuario.py
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from interfaz_usuario import main as iniciar_interfaz
        
        # Iniciar interfaz
        iniciar_interfaz()
        
    except Exception as e:
        print(f"\nError al iniciar la interfaz de usuario: {e}")
        print("Ejecutando fallback a consola...")
        
        # Ofrecer modo consola
        respuesta = input("\n¿Desea ejecutar el pipeline en modo consola? (s/n): ")
        if respuesta.lower() in ['s', 'si', 'sí', 'y', 'yes']:
            ejecutar_modo_consola(corpus_dir, resultados_dir)
        else:
            print("\nSaliendo...")

if __name__ == "__main__":
    main()
