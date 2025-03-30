import os
import pdfplumber
from pathlib import Path

def pdf_to_text(pdf_folder, output_folder):
    """
    Convierte todos los archivos PDF en una carpeta a archivos de texto sin perder información.
    
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
                text = "\n\n".join(page.extract_text() or '' for page in pdf.pages)
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            
            print(f"Convertido: {pdf_path} -> {output_path}")
        except Exception as e:
            print(f"Error al procesar {pdf_path}: {e}")
    
    print(f"Conversión completa. {len(pdf_files)} archivos procesados.")

# Uso del script
if __name__ == "__main__":
    pdf_folder = "corpus"  # Cambia esto por la ruta a tu carpeta de PDFs
    output_folder = "corpus_txt"  # Cambia esto por la ruta donde quieres guardar los TXTs
    pdf_to_text(pdf_folder, output_folder)
