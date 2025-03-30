#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para corregir problemas de codificación en el pipeline de análisis de becas.
Este script debe modificar los archivos principales para asegurar que manejen correctamente
codificaciones UTF-8 en archivos temporales.
"""

import os
import sys
import re
from pathlib import Path

def corregir_archivo(ruta_archivo):
    """
    Corrige los problemas de codificación en un archivo específico.
    
    Args:
        ruta_archivo: Ruta al archivo a modificar
    
    Returns:
        bool: True si se realizaron cambios, False en caso contrario
    """
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
        
        # Verificar si ya tiene la declaración de codificación
        if '# -*- coding: utf-8 -*-' in contenido:
            print(f"El archivo {ruta_archivo} ya tiene declaración de codificación UTF-8")
            
            # Aún así verificamos el manejo de archivos temporales
            contenido_modificado = contenido
            
            # Modificar creación de archivos temporales para especificar encoding='utf-8'
            contenido_modificado = re.sub(
                r'with tempfile\.NamedTemporaryFile\(suffix=\'\.py\', delete=False, mode=\'w\'\)',
                r'with tempfile.NamedTemporaryFile(suffix=\'.py\', delete=False, mode=\'w\', encoding=\'utf-8\')',
                contenido_modificado
            )
            
            # También verificar otros patrones de apertura de archivos
            contenido_modificado = re.sub(
                r'open\(([^,]+), \'w\'\)',
                r'open(\1, \'w\', encoding=\'utf-8\')',
                contenido_modificado
            )
            
            if contenido != contenido_modificado:
                with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
                    archivo.write(contenido_modificado)
                print(f"Se modificó el manejo de archivos en {ruta_archivo}")
                return True
            else:
                print(f"No fue necesario modificar {ruta_archivo}")
                return False
        
        # Si no tiene declaración de codificación, añadirla al principio
        lineas = contenido.split('\n')
        nueva_primera_linea = '#!/usr/bin/env python'
        nueva_segunda_linea = '# -*- coding: utf-8 -*-'
        
        if lineas and lineas[0].startswith('#!'):
            lineas.insert(1, nueva_segunda_linea)
        else:
            lineas.insert(0, nueva_primera_linea)
            lineas.insert(1, nueva_segunda_linea)
        
        nuevo_contenido = '\n'.join(lineas)
        
        # También modificar creación de archivos temporales
        nuevo_contenido = re.sub(
            r'with tempfile\.NamedTemporaryFile\(suffix=\'\.py\', delete=False, mode=\'w\'\)',
            r'with tempfile.NamedTemporaryFile(suffix=\'.py\', delete=False, mode=\'w\', encoding=\'utf-8\')',
            nuevo_contenido
        )
        
        # También verificar otros patrones de apertura de archivos
        nuevo_contenido = re.sub(
            r'open\(([^,]+), \'w\'\)',
            r'open(\1, \'w\', encoding=\'utf-8\')',
            nuevo_contenido
        )
        
        with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(nuevo_contenido)
        
        print(f"Se añadió la declaración de codificación UTF-8 y se modificó el manejo de archivos en {ruta_archivo}")
        return True
    
    except Exception as e:
        print(f"Error al procesar {ruta_archivo}: {e}")
        return False

def corregir_interfaz_usuario():
    """
    Corrige específicamente el archivo interfaz_usuario.py que maneja los archivos temporales
    """
    try:
        ruta_archivo = "interfaz_usuario.py"
        if not os.path.exists(ruta_archivo):
            print(f"El archivo {ruta_archivo} no existe")
            return False
        
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
        
        # Reemplazamos la creación de archivos temporales para usar encoding='utf-8'
        contenido_modificado = re.sub(
            r'with tempfile\.NamedTemporaryFile\(suffix=\'\.py\', delete=False, mode=\'w\'\)',
            r'with tempfile.NamedTemporaryFile(suffix=\'.py\', delete=False, mode=\'w\', encoding=\'utf-8\')',
            contenido
        )
        
        # Modificamos el uso de format() para texto literal que podría contener acentos
        # Buscamos patrones como .format(transformer_model=x, ...) y añadimos el encoding
        contenido_modificado = re.sub(
            r'\.format\((\s*transformer_model=.*?)\)',
            r'.format(\1)\n# Aseguramos codificación UTF-8',
            contenido_modificado
        )
        
        if contenido != contenido_modificado:
            with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
                archivo.write(contenido_modificado)
            print(f"Se actualizó la gestión de codificación en {ruta_archivo}")
            return True
        else:
            print(f"No fue necesario modificar {ruta_archivo}")
            return False
    
    except Exception as e:
        print(f"Error al procesar interfaz_usuario.py: {e}")
        return False

def corregir_codificacion_pipeline():
    """Función principal para corregir problemas de codificación en los archivos del pipeline"""
    archivos_a_corregir = [
        "ejecutar.py", 
        "interfaz_usuario.py", 
        "transformer_topic_modeling.py", 
        "summarization_models.py", 
        "llm_evaluator.py"
    ]
    
    correcciones_realizadas = False
    
    for archivo in archivos_a_corregir:
        if os.path.exists(archivo):
            if corregir_archivo(archivo):
                correcciones_realizadas = True
    
    # Corrección especial para la generación de scripts temporales en interfaz_usuario.py
    if os.path.exists("interfaz_usuario.py"):
        if corregir_interfaz_usuario():
            correcciones_realizadas = True
    
    if correcciones_realizadas:
        print("\nSe han aplicado correcciones para solucionar el problema de codificación UTF-8.")
        print("Por favor, intenta ejecutar el pipeline nuevamente.")
    else:
        print("\nNo fue necesario realizar correcciones en los archivos del pipeline.")
        print("El problema puede estar en otro lugar. Verifica que no existan caracteres especiales")
        print("en las rutas de los archivos o en los nombres de archivo.")

    # Verificar si estamos en Windows
    if sys.platform == 'win32':
        print("\nConsejo para Windows: Puedes intentar ejecutar el script con el siguiente comando:")
        print("python -X utf8 ejecutar.py")
        print("Esto fuerza a Python a usar UTF-8 como la codificación predeterminada en Windows.")

if __name__ == "__main__":
    print("=" * 60)
    print("Corrector de problemas de codificación UTF-8 para el pipeline de análisis de becas")
    print("=" * 60)
    corregir_codificacion_pipeline()