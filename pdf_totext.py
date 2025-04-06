import os
import pdfplumber
import re
from pathlib import Path

def pdf_to_text(pdf_folder, output_folder):
    """
    Convierte todos los archivos PDF en una carpeta a archivos de texto, eliminando
    información de CSV, direcciones de validación y firmantes.
    
    Args:
        pdf_folder: Ruta a la carpeta que contiene los archivos PDF.
        output_folder: Ruta a la carpeta donde se guardarán los archivos de texto.
    """
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = list(Path(pdf_folder).glob('*.pdf'))
    
    for pdf_path in pdf_files:
        file_name = pdf_path.stem
        output_path = os.path.join(output_folder, f"{file_name}.txt")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extraer texto de todas las páginas
                text = "\n\n".join(page.extract_text() or '' for page in pdf.pages)
                
                # Limpiar el texto
                cleaned_text = clean_text(text)
                
                # Guardar el texto limpio
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(cleaned_text)
                
                print(f"Convertido: {pdf_path} -> {output_path}")
        except Exception as e:
            print(f"Error al procesar {pdf_path}: {e}")
    
    print(f"Conversión completa. {len(pdf_files)} archivos procesados.")

def clean_text(text):
    """
    Elimina las líneas específicas que dificultan el procesamiento.
    
    Args:
        text: El texto extraído del PDF.
    
    Returns:
        Texto limpio sin las líneas problemáticas.
    """
    # Patrones para eliminar
    patterns = [
        # Patrones originales
        r'CSV\s*:\s*GEN-[a-zA-Z0-9-]+',
        r'DIRECCIÓN DE VALIDACIÓN\s*:.*?\.htm',
        r'FIRMANTE\(\d+\)\s*:.*?NOTAS\s*:\s*F',
        r'\d+\s*ALCALÁ,\s*\d+\s*\d+\s*MADRID',
        
        # Nuevos patrones
        r'FIRMANTE\(\d+\)\s*:.*?Aprueba',  # Nuevo formato de firmante
        r'FIRMANTE\(\d+\)\s*:.*?(FECHA|DATE).*',  # Cualquier línea de firmante
        
        # Texto invertido (normalmente dirección de validación invertida)
        r'\.{3}tlusnoc/soicivres/tnorFedeSgap.*',
        r'nóiccerid\s+etneiugis\s+al\s+ne\s+otnemucod\s+etse.*',
        r'dadirgetni\s+al\s+racifirev\s+edeuP.*',
        r'nóicacifireV\s+ed\s+oruges\s+ogidóC.*',
        
        # Patrones genéricos para códigos de verificación (tanto normal como invertido)
        r'([A-Z]+-[a-zA-Z0-9-]+\s*:\s*.*?edeuP|.*?edeuP\s*:\s*[A-Z]+-[a-zA-Z0-9-]+)'
    ]
    
    # Aplicar patrones de limpieza
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Eliminar líneas que contienen fragmentos específicos
    lines_to_remove = [
        "Código seguro de Verificación",
        "Puede verificar la integridad",
        "DIRECCIÓN DE VALIDACIÓN",
        "CSV :",
        "FIRMANTE",
        "https://sede.administracion.gob.es"
    ]
    
    # Dividir en líneas, filtrar y volver a unir
    cleaned_lines = []
    for line in cleaned_text.split('\n'):
        if not any(fragment in line for fragment in lines_to_remove):
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Eliminar líneas en blanco múltiples
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text

# Uso del script
if __name__ == "__main__":
    pdf_folder = "corpus"  # Cambia esto por la ruta a tu carpeta de PDFs
    output_folder = "corpus_txt"  # Cambia esto por la ruta donde quieres guardar los TXTs
    pdf_to_text(pdf_folder, output_folder)