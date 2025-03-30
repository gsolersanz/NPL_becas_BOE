# Analizador de Documentos de Becas

Este es un sistema completo para analizar documentos de becas, extraer información relevante, generar resúmenes automáticos y evaluar la calidad de dichos resúmenes.

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Instalación](#instalación)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Cómo Utilizar la Aplicación](#cómo-utilizar-la-aplicación)
6. [Flujo de Funcionamiento](#flujo-de-funcionamiento)
7. [Componentes Principales](#componentes-principales)
8. [Solución de Problemas](#solución-de-problemas)
9. [Funcionalidades Avanzadas](#funcionalidades-avanzadas)

## Descripción General

Este sistema utiliza técnicas avanzadas de procesamiento de lenguaje natural (NLP) para analizar documentos de becas, identificar temas clave, extraer información relevante de diferentes aspectos (como requisitos económicos, académicos, plazos, etc.), generar resúmenes automáticos de estos documentos y evaluar dichos resúmenes con modelos de lenguaje.

## Requisitos del Sistema

- Python 3.8 o superior
- Dependencias principales:
  - transformers
  - torch
  - sentence_transformers
  - scikit-learn
  - nltk
  - pandas
  - matplotlib
  - seaborn
  - tqdm
  - customtkinter (opcional, para mejorar la interfaz)

## Instalación

1. **Clona o descarga** este repositorio a tu máquina local
2. **Ejecuta el script principal**, que verificará automáticamente las dependencias:
   ```
   python ejecutar.py
   ```
   
El script te preguntará si deseas instalar las dependencias faltantes automáticamente.

## Estructura del Proyecto

El proyecto está organizado en los siguientes archivos principales:

- `ejecutar.py`: Script principal que inicia la aplicación, verifica dependencias y orquesta el flujo de trabajo
- `interfaz_usuario.py`: Interfaz gráfica para interactuar con el sistema
- `transformer_topic_modeling.py`: Módulo para el análisis de temas y extracción de información
- `summarization_models.py`: Módulo para la generación de resúmenes automáticos
- `llm_evaluator.py`: Módulo para evaluar la calidad de los resúmenes generados

## Cómo Utilizar la Aplicación

La aplicación tiene una interfaz gráfica con cuatro pestañas principales:

### 1. Selección de Archivos

- Selecciona la carpeta donde están tus documentos de becas (archivos .txt)
- Especifica la carpeta donde quieres guardar los resultados
- Selecciona los archivos que deseas analizar

### 2. Configuración

- **Análisis de Temas**: 
  - Elige el número de temas (clusters) a identificar
  - Selecciona el modelo Transformer para análisis semántico

- **Modelos de Resumen**:
  - Selecciona qué modelos de resumen automático quieres utilizar
  - Opciones: BART, T5, PEGASUS, LED, ProphetNet

- **Modelo Evaluador**:
  - Elige el modelo LLM para evaluar los resúmenes
  - Opciones: GPT-2, FLAN-T5, Llama-2

### 3. Análisis de Temas

- **Selección Manual de Temas**: 
  - Elige un archivo específico
  - Selecciona un campo semántico concreto (requisitos económicos, requisitos académicos, etc.)
  - Extrae información relevante sobre ese campo

- **Temas Identificados Automáticamente**:
  - Muestra los temas identificados automáticamente
  - Te permite seleccionar los temas de interés

- **Archivos por Tema**:
  - Muestra los archivos generados para cada tema
  - Te permite seleccionar archivos específicos para resumen

### 4. Resultados

- Muestra los resúmenes generados y evaluados
- Indica cuál es el mejor modelo para cada resumen
- Proporciona la justificación de la evaluación
- Permite acceder directamente a la carpeta de resultados

## Flujo de Funcionamiento

La aplicación sigue el siguiente flujo de trabajo:

1. **Selección y Verificación**: Se seleccionan los archivos a analizar y se configuran los parámetros
2. **Análisis de Temas**: 
   - Se utilizan modelos de embeddings para analizar el texto
   - Se agrupan párrafos similares en clusters (temas)
   - Se identifican los temas principales de cada documento
3. **Extracción de Información**:
   - Se extraen secciones relevantes para cada tema
   - Se organizan en archivos separados
4. **Generación de Resúmenes**:
   - Se procesan las secciones relevantes con múltiples modelos
   - Cada modelo genera un resumen diferente
5. **Evaluación de Resúmenes**:
   - Se analizan los resúmenes con un modelo LLM
   - Se determina cuál es el mejor resumen
6. **Presentación de Resultados**:
   - Se muestran los mejores resúmenes con su evaluación

## Componentes Principales

### Análisis de Temas (transformer_topic_modeling.py)

Este módulo implementa un sistema de análisis temático basado en embeddings y clustering:

- Utiliza modelos de Sentence Transformers para generar embeddings de texto
- Implementa K-means para agrupar párrafos similares
- Mapea automáticamente clusters a categorías predefinidas
- Extrae secciones relevantes para cada tema
- Ofrece un método manual para extraer información sobre un campo semántico específico

Categorías predefinidas:
- `documentación_y_plazos`
- `requisitos_económicos` 
- `requisitos_académicos`
- `cuantías_y_ayudas`
- `procedimiento_resolución`

### Generación de Resúmenes (summarization_models.py)

Este módulo gestiona múltiples modelos de resumen automático:

- BART: Equilibrio entre velocidad y calidad
- T5: Rápido y versátil
- PEGASUS: Especializado en resúmenes de alta calidad
- LED: Para documentos muy largos
- ProphetNet: Para resúmenes coherentes

### Evaluación de Resúmenes (llm_evaluator.py)

Este módulo evalúa la calidad de los resúmenes utilizando modelos de lenguaje:

- Analiza aspectos como cobertura, concisión y coherencia
- Usa GPT-2, FLAN-T5 o Llama-2 para evaluación
- Proporciona justificación detallada sobre el mejor resumen
- Incluye mecanismos de respaldo por si falla el modelo principal

## Solución de Problemas

### Errores de codificación UTF-8
Si encuentras errores relacionados con caracteres especiales:
- Asegúrate de que todos los archivos tengan la declaración `# -*- coding: utf-8 -*-`
- En Windows, usa: `python -X utf8 ejecutar.py`

### Problemas con modelos
Si hay errores al cargar los modelos:
- El sistema implementa automáticamente alternativas más ligeras
- Verifica tu conexión a internet para descargar los modelos

### Archivos no detectados
Si tus archivos no aparecen:
- Asegúrate de que sean archivos .txt
- Verifica que estén en la carpeta seleccionada

## Funcionalidades Avanzadas

### Selección Manual de Campos Semánticos

Además del análisis automático, puedes solicitar información específica sobre:

1. **Documentación y plazos**: Información sobre qué documentos presentar y cuándo
2. **Requisitos económicos**: Umbrales de renta, patrimonio, deducciones, etc.
3. **Requisitos académicos**: Créditos, notas mínimas, rendimiento necesario
4. **Cuantías y ayudas**: Importes de becas, componentes, condiciones
5. **Procedimiento de resolución**: Información sobre concesión, denegación, recursos

### Visualización de Distribución de Temas

El sistema genera automáticamente una visualización de la distribución de temas en cada documento, guardada como `distribucion_temas.png` en la carpeta de resultados.

### Modo Consola

Si la interfaz gráfica tiene problemas, el sistema ofrece un modo de consola alternativo que realiza las mismas funciones a través de comandos en terminal.