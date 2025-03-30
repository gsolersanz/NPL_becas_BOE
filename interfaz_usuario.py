#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import subprocess
import webbrowser
import tempfile
from datetime import datetime

# Verificar dependencias
try:
    import customtkinter as ctk
    CUSTOM_TK = True
except ImportError:
    CUSTOM_TK = False
    print("customtkinter no está instalado. Se usará tkinter estándar.")
    print("Para una mejor experiencia, instala customtkinter: pip install customtkinter")

class BecasAnalyzerApp:
    def __init__(self, root):
        """
        Inicializa la aplicación.
        
        Args:
            root: Raíz de la aplicación Tkinter
        """
        self.root = root
        self.root.title("Analizador de Becas - Pipeline Completo")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        if CUSTOM_TK:
            ctk.set_appearance_mode("System")
            ctk.set_default_color_theme("blue")
        
        # Variables de estado
        self.input_folder = tk.StringVar(value="")
        self.output_folder = tk.StringVar(value="")
        self.selected_files = []
        self.selected_topics = []
        self.available_topics = []
        self.topic_files = {}
        self.summarization_models = {
            "bart": tk.BooleanVar(value=True),
            "t5": tk.BooleanVar(value=True),
            "pegasus": tk.BooleanVar(value=True),
            "led": tk.BooleanVar(value=False),  # Desactivado por defecto (lento)
            "prophetnet": tk.BooleanVar(value=False)  # Desactivado por defecto (lento)
        }
        self.llm_model = tk.StringVar(value="gpt2")
        self.num_topics = tk.IntVar(value=5)
        self.transformer_model = tk.StringVar(value="distiluse-base-multilingual-cased-v1")
        
        # Crear interfaz
        self.create_widgets()
        
        # Carpeta de salida predeterminada (subdirectorio 'resultados' en la carpeta actual)
        default_output = os.path.join(os.getcwd(), "resultados")
        self.output_folder.set(default_output)
        self.output_path_entry.delete(0, tk.END)
        self.output_path_entry.insert(0, default_output)
    
    def browse_output_folder(self):
        """Abre un diálogo para seleccionar la carpeta de salida."""
        folder = filedialog.askdirectory(title="Selecciona la carpeta de resultados")
        if folder:
            self.output_folder.set(folder)
            self.output_path_entry.delete(0, tk.END)
            self.output_path_entry.insert(0, folder)

    def create_widgets(self):
        """Crea los widgets de la interfaz."""
        # Panel principal con pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Selección de Archivos
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="1. Selección de Archivos")
        
        # Pestaña 2: Configuración de Modelos
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="2. Configuración")
        
        # Pestaña 3: Análisis de Temas
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="3. Análisis de Temas")
        
        # Pestaña 4: Resultados
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="4. Resultados")
        
        # Contenido de la Pestaña 1: Selección de Archivos
        self.setup_file_selection_tab()
        
        # Contenido de la Pestaña 2: Configuración de Modelos
        self.setup_models_configuration_tab()
        
        # Contenido de la Pestaña 3: Análisis de Temas
        self.setup_topic_analysis_tab()
        
        # Contenido de la Pestaña 4: Resultados
        self.setup_results_tab()
        
        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="Listo", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.progress.pack_forget()  # Ocultar inicialmente
    
    def setup_file_selection_tab(self):
        """Configura la pestaña de selección de archivos."""
        frame = ttk.Frame(self.tab1, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Carpeta de entrada
        ttk.Label(frame, text="Carpeta de documentos:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        input_frame = ttk.Frame(frame)
        input_frame.grid(row=0, column=1, sticky=tk.EW, pady=5)
        
        self.input_path_entry = ttk.Entry(input_frame, textvariable=self.input_folder, width=50)
        self.input_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(input_frame, text="Examinar", command=self.browse_input_folder)
        browse_btn.pack(side=tk.RIGHT, padx=5)
        
        # Carpeta de salida
        ttk.Label(frame, text="Carpeta de resultados:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        output_frame = ttk.Frame(frame)
        output_frame.grid(row=1, column=1, sticky=tk.EW, pady=5)
        
        self.output_path_entry = ttk.Entry(output_frame, textvariable=self.output_folder, width=50)
        self.output_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_output_btn = ttk.Button(output_frame, text="Examinar", command=self.browse_output_folder)
        browse_output_btn.pack(side=tk.RIGHT, padx=5)
        
        # Lista de archivos
        ttk.Label(frame, text="Archivos disponibles:").grid(row=2, column=0, sticky=tk.NW, pady=5)
        
        files_frame = ttk.Frame(frame)
        files_frame.grid(row=2, column=1, sticky=tk.NSEW, pady=5)
        
        self.files_listbox = tk.Listbox(files_frame, selectmode=tk.MULTIPLE, height=10)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=files_scrollbar.set)
        
        # Botones de acción
        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(row=3, column=1, sticky=tk.E, pady=10)
        
        refresh_btn = ttk.Button(buttons_frame, text="Actualizar archivos", command=self.refresh_files)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        select_all_btn = ttk.Button(buttons_frame, text="Seleccionar todos", command=self.select_all_files)
        select_all_btn.pack(side=tk.LEFT, padx=5)
        
        deselect_all_btn = ttk.Button(buttons_frame, text="Deseleccionar todos", command=self.deselect_all_files)
        deselect_all_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón para continuar
        next_btn = ttk.Button(frame, text="Siguiente →", command=lambda: self.notebook.select(1))
        next_btn.grid(row=4, column=1, sticky=tk.E, pady=10)
        
        # Hacer que la columna se expanda
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
    
    def setup_models_configuration_tab(self):
        """Configura la pestaña de configuración de modelos."""
        frame = ttk.Frame(self.tab2, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Sección: Configuración del Topic Modeling
        topic_frame = ttk.LabelFrame(frame, text="Configuración del Análisis de Temas")
        topic_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(topic_frame, text="Número de temas:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(topic_frame, from_=2, to=10, textvariable=self.num_topics, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(topic_frame, text="Modelo Transformer:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        transformer_models = [
            "distiluse-base-multilingual-cased-v1",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "multi-qa-mpnet-base-dot-v1"
        ]
        
        transformer_combo = ttk.Combobox(topic_frame, textvariable=self.transformer_model, values=transformer_models, width=40)
        transformer_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Sección: Modelos de Resumen
        summary_frame = ttk.LabelFrame(frame, text="Modelos de Resumen")
        summary_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(summary_frame, text="Selecciona los modelos a utilizar:").grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        row = 1
        for model, var in self.summarization_models.items():
            description = {
                "bart": "BART (rápido, buen equilibrio)",
                "t5": "T5 (rápido, versátil)",
                "pegasus": "PEGASUS (especializado en resúmenes)",
                "led": "LED (para documentos largos, lento)",
                "prophetnet": "ProphetNet (resúmenes coherentes, lento)"
            }
            
            ttk.Checkbutton(summary_frame, text=description.get(model, model), variable=var).grid(row=row, column=0, sticky=tk.W, padx=20, pady=2)
            row += 1
        
        # Sección: Modelo Evaluador (LLM)
        evaluator_frame = ttk.LabelFrame(frame, text="Modelo Evaluador (LLM)")
        evaluator_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(evaluator_frame, text="Modelo para evaluar resúmenes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        llm_models = [
            ("gpt2", "GPT-2 (rápido, ligero)"),
            ("flan-t5", "FLAN-T5 (mejor calidad, más lento)"),
            ("llama2", "Llama-2 (mejor calidad, requiere GPU potente)")
        ]
        
        for i, (model_id, model_desc) in enumerate(llm_models):
            ttk.Radiobutton(evaluator_frame, text=model_desc, value=model_id, variable=self.llm_model).grid(row=i+1, column=0, sticky=tk.W, padx=20, pady=2)
        
        # Botones de navegación
        nav_frame = ttk.Frame(frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        back_btn = ttk.Button(nav_frame, text="← Atrás", command=lambda: self.notebook.select(0))
        back_btn.pack(side=tk.LEFT)
        
        next_btn = ttk.Button(nav_frame, text="Siguiente →", command=self.start_topic_analysis)
        next_btn.pack(side=tk.RIGHT)
    
    def setup_topic_analysis_tab(self):
        """Configura la pestaña de análisis de temas."""
        frame = ttk.Frame(self.tab3, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial
        self.topic_message = ttk.Label(frame, text="Para ver los temas, primero ejecuta el análisis desde la pestaña anterior.")
        self.topic_message.pack(pady=20)
        
        # Panel dividido
        self.topic_paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        
        # Panel izquierdo (selección manual de temas)
        manual_frame = ttk.LabelFrame(self.topic_paned, text="Selección Manual de Temas")
        
        # Lista de campos semánticos predefinidos
        self.campos_semanticos = [
            "documentación_y_plazos",
            "requisitos_económicos", 
            "requisitos_académicos", 
            "cuantías_y_ayudas", 
            "procedimiento_resolución"
        ]
        
        # Etiqueta explicativa
        ttk.Label(manual_frame, text="Selecciona un campo semántico\npara extraer información relevante:", 
                justify=tk.LEFT).pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)
        
        # Variable para almacenar la selección
        self.selected_semantic_field = tk.StringVar(value=self.campos_semanticos[0])
        
        # Opciones de campos semánticos
        for campo in self.campos_semanticos:
            campo_display = campo.replace('_', ' ').capitalize()
            ttk.Radiobutton(manual_frame, text=campo_display, value=campo, 
                        variable=self.selected_semantic_field).pack(side=tk.TOP, padx=20, pady=2, anchor=tk.W)
        
        # Selección del archivo para análisis manual
        ttk.Label(manual_frame, text="Selecciona un archivo:", justify=tk.LEFT).pack(
            side=tk.TOP, padx=5, pady=5, anchor=tk.W)
        
        # Lista de archivos para análisis manual
        self.manual_files_listbox = tk.Listbox(manual_frame, height=6)
        self.manual_files_listbox.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Botón para extraer información manualmente
        extract_btn = ttk.Button(manual_frame, text="Extraer Información", 
                            command=self.extract_semantic_field)
        extract_btn.pack(side=tk.TOP, padx=5, pady=10)
        
        # Lista de temas (los que se identifiquen automáticamente)
        topics_frame = ttk.LabelFrame(self.topic_paned, text="Temas Identificados Automáticamente")
        
        self.topics_listbox = tk.Listbox(topics_frame, selectmode=tk.MULTIPLE, height=15)
        self.topics_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        topics_scrollbar = ttk.Scrollbar(topics_frame, orient=tk.VERTICAL, command=self.topics_listbox.yview)
        topics_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.topics_listbox.config(yscrollcommand=topics_scrollbar.set)
        
        # Lista de archivos de temas
        files_frame = ttk.LabelFrame(self.topic_paned, text="Archivos por Tema")
        
        self.topic_files_listbox = tk.Listbox(files_frame, selectmode=tk.MULTIPLE, height=15)
        self.topic_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.topic_files_listbox.yview)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.topic_files_listbox.config(yscrollcommand=files_scrollbar.set)
        
        # Añadir frames al panel
        self.topic_paned.add(manual_frame, weight=1)
        self.topic_paned.add(topics_frame, weight=1)
        self.topic_paned.add(files_frame, weight=2)
        
        # Botones de acción
        action_frame = ttk.Frame(frame)
        
        select_all_topics_btn = ttk.Button(action_frame, text="Seleccionar todos los temas", command=self.select_all_topics)
        select_all_topics_btn.pack(side=tk.LEFT, padx=5)
        
        select_all_files_btn = ttk.Button(action_frame, text="Seleccionar todos los archivos", command=self.select_all_topic_files)
        select_all_files_btn.pack(side=tk.LEFT, padx=5)
        
        # Botones de navegación
        nav_frame = ttk.Frame(frame)
        
        back_btn = ttk.Button(nav_frame, text="← Atrás", command=lambda: self.notebook.select(1))
        back_btn.pack(side=tk.LEFT, padx=5)
        
        generate_btn = ttk.Button(nav_frame, text="Generar Resúmenes", command=self.start_summarization)
        generate_btn.pack(side=tk.RIGHT, padx=5)
        
        # Inicialmente ocultar el panel dividido y los botones
        self.topic_paned.pack_forget()
        action_frame.pack_forget()
        nav_frame.pack_forget()
        
        # Guardar referencias para usar más tarde
        self.topic_action_frame = action_frame
        self.topic_nav_frame = nav_frame

    def browse_input_folder(self):
        """Abre un diálogo para seleccionar la carpeta de entrada."""
        folder = filedialog.askdirectory(title="Selecciona la carpeta de documentos")
        if folder:
            self.input_folder.set(folder)
            self.input_path_entry.delete(0, tk.END)
            self.input_path_entry.insert(0, folder)
            self.refresh_files()


    def extract_semantic_field(self):
        """Extrae información relevante para el campo semántico seleccionado."""
        # Verificar que hay archivos seleccionados
        selected_indices = self.manual_files_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Advertencia", "Selecciona un archivo para extraer información.")
            return
        
        # Obtener archivo seleccionado
        idx = selected_indices[0]
        file_name = self.manual_files_listbox.get(idx)
        input_path = self.input_folder.get()
        file_path = os.path.join(input_path, file_name)
        
        # Obtener campo semántico seleccionado
        campo_semantico = self.selected_semantic_field.get()
        
        # Mostrar progreso
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.progress.start()
        self.status_bar.config(text=f"Extrayendo información sobre {campo_semantico}...")
        
        # Ejecutar extracción en un hilo separado para no bloquear la interfaz
        thread = threading.Thread(target=self._run_semantic_extraction, 
                                args=(file_path, campo_semantico))
        thread.daemon = True
        thread.start()

    def _run_semantic_extraction(self, file_path, campo_semantico):
        """Ejecuta la extracción semántica en un hilo separado."""
        try:
            # Crear carpeta de salida
            output_folder = self.output_folder.get()
            topics_folder = os.path.join(output_folder, "1_temas")
            os.makedirs(topics_folder, exist_ok=True)
            
            # Leer el archivo
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Nombre base del archivo
            base_name = os.path.basename(file_path).split('.')[0]
            
            # Crear un script temporal para ejecutar la extracción
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_file.write("""#!/usr/bin/env python
    # -*- coding: utf-8 -*-

    import sys
    import os

    # Asegurarse de que el directorio actual está en el path
    sys.path.append(os.getcwd())

    # Importar modelo de extracción
    try:
        from transformer_topic_modeling import BecasTransformerTopicModel
        
        # Crear instancia del modelo
        modelo = BecasTransformerTopicModel()
        
        # Leer el archivo
        with open("{file_path}", 'r', encoding='utf-8') as f:
            texto = f.read()
        
        # Extraer información relevante
        campo_semantico = "{campo_semantico}"
        resultado = modelo.extract_by_category(texto, campo_semantico)
        
        # Guardar resultado
        output_file = "{output_file}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Documento: {base_name}\\n")
            f.write("Campo semántico: {campo_display}\\n\\n")
            f.write(resultado)
        
        print("Extracción completada correctamente")
        
    except Exception as e:
        print(f"Error en la extracción: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    """.format(
        file_path=file_path.replace('\\', '\\\\'),
        campo_semantico=campo_semantico,
        output_file=os.path.join(topics_folder, f"{base_name}_{campo_semantico}.txt").replace('\\', '\\\\'),
        base_name=base_name,
        campo_display=campo_semantico.replace('_', ' ').capitalize()
    ))
                script_path = temp_file.name
            
            # Ejecutar script
            result = subprocess.run([sys.executable, script_path], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                text=True)
            
            # Eliminar script temporal
            try:
                os.unlink(script_path)
            except:
                pass
            
            # Verificar resultado
            if result.returncode != 0:
                self.root.after(0, lambda: messagebox.showerror("Error", 
                                                            f"Error en la extracción semántica:\n{result.stderr}"))
                self.root.after(0, self._finalize_extraction_error)
                return
            
            # Actualizar interfaz en el hilo principal
            self.root.after(0, lambda: self._update_manual_extraction_ui(base_name, campo_semantico))
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error inesperado: {str(e)}"))
            self.root.after(0, self._finalize_extraction_error)

def _update_manual_extraction_ui(self, base_name, campo_semantico):
    """Actualiza la interfaz tras la extracción manual."""
    # Detener indicador de progreso
    self.progress.stop()
    self.progress.pack_forget()
    
    # Actualizar tema en la lista de archivos
    output_folder = self.output_folder.get()
    topics_folder = os.path.join(output_folder, "1_temas")
    
    # Añadir a la lista de archivos de temas
    safe_topic = campo_semantico.replace('_', '-')
    filename = f"{base_name}_{safe_topic}.txt"
    
    # Limpiar selección actual
    self.topic_files_listbox.selection_clear(0, tk.END)
    
    # Añadir el nuevo archivo al final
    self.topic_files_listbox.insert(tk.END, filename)
    
    # Seleccionar el nuevo archivo
    last_index = self.topic_files_listbox.size() - 1
    self.topic_files_listbox.selection_set(last_index)
    self.topic_files_listbox.see(last_index)
    
    # Actualizar estado
    campo_display = campo_semantico.replace('_', ' ').capitalize()
    self.status_bar.config(text=f"Extracción completada: {campo_display} de {base_name}")
    
    # Mostrar mensaje de éxito
    messagebox.showinfo("Extracción Completada", 
                      f"Se ha extraído información sobre '{campo_display}' del documento '{base_name}'.\n\n"
                      f"El archivo resultante se ha añadido a la lista de archivos de temas.")

def _finalize_extraction_error(self):
    """Finaliza la extracción con error."""
    self.progress.stop()
    self.progress.pack_forget()
    self.status_bar.config(text="Error en la extracción semántica")

def _update_topic_ui(self):
    """Actualiza la interfaz después del análisis de temas."""
    # Ocultar mensaje inicial
    self.topic_message.pack_forget()
    
    # Mostrar panel dividido
    self.topic_paned.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Limpiar listas
    self.topics_listbox.delete(0, tk.END)
    self.topic_files_listbox.delete(0, tk.END)
    self.manual_files_listbox.delete(0, tk.END)
    
    # Mostrar temas disponibles
    for topic in self.available_topics:
        # Formatear nombre del tema para mostrar
        display_name = topic.replace('_', ' ').capitalize()
        self.topics_listbox.insert(tk.END, display_name)
    
    # Añadir archivos a la lista de selección manual
    input_path = self.input_folder.get()
    for file_path in self.selected_files:
        file_name = os.path.basename(file_path)
        self.manual_files_listbox.insert(tk.END, file_name)
    
    # Configurar selección de temas
    self.topics_listbox.bind('<<ListboxSelect>>', self.on_topic_selected)
    
    # Mostrar botones
    self.topic_action_frame.pack(pady=10)
    self.topic_nav_frame.pack(fill=tk.X, pady=10)
    
    # Mover a la siguiente pestaña
    self.notebook.tab(1, state="normal")
    self.notebook.select(2)
    
    # Detener indicador de progreso
    self.progress.stop()
    self.progress.pack_forget()
    self.status_bar.config(text="Análisis de temas completado")
    
    def setup_results_tab(self):
        """Configura la pestaña de resultados."""
        frame = ttk.Frame(self.tab4, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial
        self.results_message = ttk.Label(frame, text="Para ver los resultados, primero genera los resúmenes desde la pestaña anterior.")
        self.results_message.pack(pady=20)
        
        # Panel de resultados (inicialmente oculto)
        self.results_frame = ttk.Frame(frame)
        
        # Lista de archivos evaluados
        files_frame = ttk.LabelFrame(self.results_frame, text="Archivos Evaluados")
        files_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.eval_files_listbox = tk.Listbox(files_frame, height=15)
        self.eval_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.eval_files_listbox.yview)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.eval_files_listbox.config(yscrollcommand=files_scrollbar.set)
        
        # Panel de detalles
        details_frame = ttk.LabelFrame(self.results_frame, text="Detalles del Mejor Resumen")
        details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Título
        self.result_title = ttk.Label(details_frame, text="", font=("TkDefaultFont", 12, "bold"))
        self.result_title.pack(fill=tk.X, padx=5, pady=5)
        
        # Mejor modelo
        model_frame = ttk.Frame(details_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Mejor modelo:").pack(side=tk.LEFT)
        self.best_model_label = ttk.Label(model_frame, text="", font=("TkDefaultFont", 10, "bold"))
        self.best_model_label.pack(side=tk.LEFT, padx=5)
        
        # Texto del resumen
        ttk.Label(details_frame, text="Resumen:").pack(anchor=tk.W, padx=5)
        
        summary_frame = ttk.Frame(details_frame)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_text = tk.Text(summary_frame, wrap=tk.WORD, height=10)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.config(yscrollcommand=summary_scrollbar.set)
        
        # Razonamiento del LLM
        ttk.Label(details_frame, text="Justificación del evaluador LLM:").pack(anchor=tk.W, padx=5)
        
        reasoning_frame = ttk.Frame(details_frame)
        reasoning_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.reasoning_text = tk.Text(reasoning_frame, wrap=tk.WORD, height=8)
        self.reasoning_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        reasoning_scrollbar = ttk.Scrollbar(reasoning_frame, orient=tk.VERTICAL, command=self.reasoning_text.yview)
        reasoning_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.reasoning_text.config(yscrollcommand=reasoning_scrollbar.set)
        
        # Botones de acción
        action_frame = ttk.Frame(frame)
        
        back_btn = ttk.Button(action_frame, text="← Atrás", command=lambda: self.notebook.select(2))
        back_btn.pack(side=tk.LEFT, padx=5)
        
        open_folder_btn = ttk.Button(action_frame, text="Abrir Carpeta de Resultados", command=self.open_results_folder)
        open_folder_btn.pack(side=tk.RIGHT, padx=5)
        
        # Inicialmente ocultar
        self.results_frame.pack_forget()
        action_frame.pack_forget()
        
        # Guardar referencia
        self.results_action_frame = action_frame
    
    def browse_input_folder(self):
        """Abre un diálogo para seleccionar la carpeta de entrada."""
        folder = filedialog.askdirectory(title="Selecciona la carpeta de documentos")
        if folder:
            self.input_folder.set(folder)
            self.input_path_entry.delete(0, tk.END)
            self.input_path_entry.insert(0, folder)
            self.refresh_files()
    
    def browse_output_folder(self):
        """Abre un diálogo para seleccionar la carpeta de salida."""
        folder = filedialog.askdirectory(title="Selecciona la carpeta de resultados")
        if folder:
            self.output_folder.set(folder)
            self.output_path_entry.delete(0, tk.END)
            self.output_path_entry.insert(0, folder)
    
    def refresh_files(self):
        """Actualiza la lista de archivos disponibles."""
        input_path = self.input_folder.get()
        if not input_path or not os.path.isdir(input_path):
            messagebox.showwarning("Advertencia", "Selecciona una carpeta de documentos válida.")
            return
        
        # Limpiar lista
        self.files_listbox.delete(0, tk.END)
        
        # Buscar archivos de texto
        files = list(Path(input_path).glob('*.txt'))
        
        if not files:
            self.files_listbox.insert(tk.END, "No se encontraron archivos de texto (.txt)")
            return
        
        # Añadir archivos a la lista
        for file in sorted(files):
            self.files_listbox.insert(tk.END, file.name)
    
    def select_all_files(self):
        """Selecciona todos los archivos de la lista."""
        self.files_listbox.select_set(0, tk.END)
    
    def deselect_all_files(self):
        """Deselecciona todos los archivos de la lista."""
        self.files_listbox.selection_clear(0, tk.END)
    
    def get_selected_files(self):
        """Obtiene los archivos seleccionados."""
        selected_indices = self.files_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Advertencia", "No hay archivos seleccionados.")
            return []
        
        selected_files = []
        input_path = self.input_folder.get()
        
        for idx in selected_indices:
            file_name = self.files_listbox.get(idx)
            file_path = os.path.join(input_path, file_name)
            selected_files.append(file_path)
        
        return selected_files
    
    def start_topic_analysis(self):
        """Inicia el análisis de temas."""
        # Verificar archivos seleccionados
        selected_files = self.get_selected_files()
        if not selected_files:
            return
        
        # Verificar carpeta de salida
        output_folder = self.output_folder.get()
        if not output_folder:
            messagebox.showwarning("Advertencia", "Selecciona una carpeta de resultados válida.")
            return
        
        # Crear carpeta de salida si no existe
        os.makedirs(output_folder, exist_ok=True)
        
        # Guardar selecciones para usar más tarde
        self.selected_files = selected_files
        
        # Mostrar progreso
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.progress.start()
        self.status_bar.config(text="Analizando temas...")
        
        # Deshabilitar pestaña durante el procesamiento
        self.notebook.tab(1, state="disabled")
        
        # Ejecutar análisis en un hilo separado
        thread = threading.Thread(target=self._run_topic_analysis)
        thread.daemon = True
        thread.start()
    
    def _run_topic_analysis(self):
        """Ejecuta el análisis de temas en un hilo separado."""
        try:
            # Crear carpetas de salida
            output_folder = self.output_folder.get()
            topics_folder = os.path.join(output_folder, "1_temas")
            os.makedirs(topics_folder, exist_ok=True)
            
            # Preparar comando
            summarization_models = [model for model, var in self.summarization_models.items() if var.get()]
            
            # Crear script temporal para ejecutar el análisis
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_file.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# Asegurarse de que el directorio actual está en el path
sys.path.append(os.getcwd())

# Importar el modelo de topic modeling
try:
    from transformer_topic_modeling import BecasTransformerTopicModel
    
    # Crear modelador de temas
    modeler = BecasTransformerTopicModel(model_name="{transformer_model}")
    
    # Cargar documentos
    for file_path in {files}:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                modeler.raw_texts.append(text)
                modeler.doc_names.append(os.path.basename(file_path).split('.')[0])
        except Exception as e:
            print(f"Error al procesar {{file_path}}: {{e}}")
    
    # Realizar análisis de temas
    results = modeler.analyze_documents(n_clusters={num_topics})
    
    # Guardar secciones por tema
    modeler.save_topic_sections("{topics_folder}")
    
    # Guardar visualización
    modeler.visualize_topics("{output_folder}/distribucion_temas.png")
    
    # Guardar información de clusters y categorías
    with open("{output_folder}/topic_mapping.txt", 'w', encoding='utf-8') as f:
        for cluster_id, category in modeler.cluster_to_category.items():
            f.write(f"{{cluster_id}}|{{category}}\\n")
    
    print("Análisis completado correctamente")
    
except Exception as e:
    print(f"Error en el análisis: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
""".format(
    transformer_model=self.transformer_model.get(),
    files=self.selected_files,
    num_topics=self.num_topics.get(),
    topics_folder=topics_folder.replace('\\', '\\\\'),
    output_folder=output_folder.replace('\\', '\\\\')
))
                script_path = temp_file.name
            
            # Ejecutar script
            result = subprocess.run([sys.executable, script_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
            
            # Eliminar script temporal
            try:
                os.unlink(script_path)
            except:
                pass
            
            # Verificar resultado
            if result.returncode != 0:
                self.root.after(0, lambda: messagebox.showerror("Error", 
                                                              f"Error en el análisis de temas:\n{result.stderr}"))
                self.root.after(0, self._finalize_topic_analysis_error)
                return
            
            # Cargar información de temas
            self.available_topics = []
            self.topic_files = {}
            
            try:
                # Cargar mapeo de clusters a categorías
                mapping_file = os.path.join(output_folder, "topic_mapping.txt")
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('|')
                            if len(parts) == 2:
                                cluster_id, category = parts
                                self.available_topics.append(category)
                
                # Buscar archivos de temas
                for topic in self.available_topics:
                    self.topic_files[topic] = []
                    for file_path in Path(topics_folder).glob(f"*{topic.replace('_', '-')}*.txt"):
                        self.topic_files[topic].append(file_path)
            
            except Exception as e:
                print(f"Error al cargar información de temas: {e}")
            
            # Actualizar interfaz en el hilo principal
            self.root.after(0, self._update_topic_ui)
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error inesperado: {str(e)}"))
            self.root.after(0, self._finalize_topic_analysis_error)
    
    def _update_topic_ui(self):
        """Actualiza la interfaz después del análisis de temas."""
        # Ocultar mensaje inicial
        self.topic_message.pack_forget()
        
        # Mostrar panel dividido
        self.topic_paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Limpiar listas
        self.topics_listbox.delete(0, tk.END)
        self.topic_files_listbox.delete(0, tk.END)
        
        # Mostrar temas disponibles
        for topic in self.available_topics:
            # Formatear nombre del tema para mostrar
            display_name = topic.replace('_', ' ').capitalize()
            self.topics_listbox.insert(tk.END, display_name)
        
        # Configurar selección de temas
        self.topics_listbox.bind('<<ListboxSelect>>', self.on_topic_selected)
        
        # Mostrar botones
        self.topic_action_frame.pack(pady=10)
        self.topic_nav_frame.pack(fill=tk.X, pady=10)
        
        # Mover a la siguiente pestaña
        self.notebook.tab(1, state="normal")
        self.notebook.select(2)
        
        # Detener indicador de progreso
        self.progress.stop()
        self.progress.pack_forget()
        self.status_bar.config(text="Análisis de temas completado")
    
    def _finalize_topic_analysis_error(self):
        """Finaliza el análisis de temas con error."""
        self.notebook.tab(1, state="normal")
        self.progress.stop()
        self.progress.pack_forget()
        self.status_bar.config(text="Error en el análisis de temas")
    
    def on_topic_selected(self, event):
        """Maneja la selección de un tema."""
        # Obtener índices seleccionados
        selected_indices = self.topics_listbox.curselection()
        if not selected_indices:
            return
        
        # Limpiar lista de archivos
        self.topic_files_listbox.delete(0, tk.END)
        
        # Mostrar archivos para los temas seleccionados
        for idx in selected_indices:
            topic = self.available_topics[idx]
            
            if topic in self.topic_files:
                # Añadir encabezado de tema
                display_topic = topic.replace('_', ' ').capitalize()
                self.topic_files_listbox.insert(tk.END, f"--- {display_topic} ---")
                
                # Añadir archivos
                for file_path in self.topic_files[topic]:
                    self.topic_files_listbox.insert(tk.END, file_path.name)
                
                # Añadir separador
                self.topic_files_listbox.insert(tk.END, "")
    
    def select_all_topics(self):
        """Selecciona todos los temas."""
        self.topics_listbox.select_set(0, tk.END)
        self.on_topic_selected(None)
    
    def select_all_topic_files(self):
        """Selecciona todos los archivos de temas."""
        self.topic_files_listbox.select_set(0, tk.END)
    
    def get_selected_topic_files(self):
        """Obtiene los archivos de tema seleccionados."""
        selected_indices = self.topic_files_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Advertencia", "No hay archivos de tema seleccionados.")
            return []
        
        # Obtener nombres de archivos seleccionados
        selected_files = []
        output_folder = self.output_folder.get()
        topics_folder = os.path.join(output_folder, "1_temas")
        
        for idx in selected_indices:
            file_name = self.topic_files_listbox.get(idx)
            
            # Ignorar encabezados y líneas vacías
            if file_name.startswith("---") or not file_name.strip():
                continue
            
            file_path = os.path.join(topics_folder, file_name)
            selected_files.append(file_path)
        
        return selected_files
    
    def start_summarization(self):
        """Inicia el proceso de generación de resúmenes."""
        # Verificar archivos seleccionados
        selected_files = self.get_selected_topic_files()
        if not selected_files:
            return
        
        # Verificar modelos seleccionados
        selected_models = [model for model, var in self.summarization_models.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("Advertencia", "Selecciona al menos un modelo de resumen.")
            return
        
        # Verificar carpeta de salida
        output_folder = self.output_folder.get()
        if not output_folder:
            messagebox.showwarning("Advertencia", "Selecciona una carpeta de resultados válida.")
            return
        
        # Mostrar progreso
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.progress.start()
        self.status_bar.config(text="Generando resúmenes...")
        
        # Deshabilitar pestaña durante el procesamiento
        self.notebook.tab(2, state="disabled")
        
        # Ejecutar generación de resúmenes en un hilo separado
        thread = threading.Thread(target=self._run_summarization, args=(selected_files, selected_models))
        thread.daemon = True
        thread.start()
    
    def _run_summarization(self, selected_files, selected_models):
        """Ejecuta la generación de resúmenes en un hilo separado."""
        try:
            # Crear carpetas de salida
            output_folder = self.output_folder.get()
            summary_folder = os.path.join(output_folder, "2_resumenes")
            eval_folder = os.path.join(output_folder, "3_evaluaciones")
            os.makedirs(summary_folder, exist_ok=True)
            os.makedirs(eval_folder, exist_ok=True)
            
            # Crear script temporal para ejecutar la generación de resúmenes
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_file.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json

# Asegurarse de que el directorio actual está en el path
sys.path.append(os.getcwd())

# Importar modelos necesarios
try:
    from summarization_models import SummarizationModels
    from llm_evaluator import ResumenevaluatorLLM
    
    # Crear instancia del resumidor
    summarizer = SummarizationModels()
    
    # Filtrar modelos seleccionados
    summarizer.models = {model: config for model, config in summarizer.models.items() 
                        if model in {selected_models}}
    
    # Procesar cada archivo
    results = []
    for file_path in {files}:
        try:
            print(f"\\nGenerando resúmenes para {{os.path.basename(file_path)}}...")
            summaries = summarizer.summarize_file(file_path, "{summary_folder}")
            
            # Crear instancia del evaluador
            evaluator = ResumenevaluatorLLM(model_name="{llm_model}")
            
            # Leer el archivo original
            with open(file_path, 'r', encoding='utf-8') as file:
                original_text = file.read()
            
            # Evaluar resúmenes
            evaluation = evaluator.evaluate_summaries(original_text, summaries)
            
            # Guardar evaluación
            base_name = os.path.basename(file_path).split('.')[0]
            eval_file = os.path.join("{eval_folder}", f"{{base_name}}_evaluacion.txt")
            with open(eval_file, 'w', encoding='utf-8') as f:
                f.write(f"EVALUACIÓN DE RESÚMENES PARA: {{base_name}}\\n\\n")
                f.write(f"MEJOR RESUMEN: {{evaluation['mejor_modelo']}}\\n\\n")
                f.write(f"RAZONAMIENTO:\\n{{evaluation['razonamiento']}}\\n\\n")
                f.write(f"EVALUACIÓN COMPLETA:\\n{{evaluation['evaluacion_completa']}}\\n\\n")
                
                # Incluir el mejor resumen al final
                if evaluation['mejor_modelo'] in summaries:
                    f.write(f"TEXTO DEL MEJOR RESUMEN ({{evaluation['mejor_modelo']}}):\\n")
                    f.write(summaries[evaluation['mejor_modelo']])
            
            # Guardar resultado
            result = {{
                "archivo": file_path,
                "nombre": base_name,
                "mejor_modelo": evaluation['mejor_modelo'],
                "mejor_resumen": summaries.get(evaluation['mejor_modelo'], ""),
                "evaluacion": evaluation['razonamiento']
            }}
            results.append(result)
            
            print(f"Mejor resumen para {{base_name}}: {{evaluation['mejor_modelo']}}")
            
        except Exception as e:
            print(f"Error al procesar {{file_path}}: {{e}}")
            import traceback
            traceback.print_exc()
    
    # Guardar resultados globales
    with open("{output_folder}/resultados.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("\\nProceso completado correctamente")
    
except Exception as e:
    print(f"Error en el proceso: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
""".format(
    selected_models=str(selected_models),
    files=str(selected_files),
    summary_folder=summary_folder.replace('\\', '\\\\'),
    eval_folder=eval_folder.replace('\\', '\\\\'),
    output_folder=output_folder.replace('\\', '\\\\'),
    llm_model=self.llm_model.get()
))
                script_path = temp_file.name
            
            # Ejecutar script
            result = subprocess.run([sys.executable, script_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
            
            # Eliminar script temporal
            try:
                os.unlink(script_path)
            except:
                pass
            
            # Verificar resultado
            if result.returncode != 0:
                self.root.after(0, lambda: messagebox.showerror("Error", 
                                                              f"Error en la generación de resúmenes:\n{result.stderr}"))
                self.root.after(0, self._finalize_summarization_error)
                return
            
            # Cargar resultados
            results_file = os.path.join(output_folder, "resultados.json")
            self.evaluation_results = []
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        self.evaluation_results = json.load(f)
                except Exception as e:
                    print(f"Error al cargar resultados: {e}")
            
            # Actualizar interfaz en el hilo principal
            self.root.after(0, self._update_results_ui)
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error inesperado: {str(e)}"))
            self.root.after(0, self._finalize_summarization_error)
    
    def _update_results_ui(self):
        """Actualiza la interfaz después de la generación de resúmenes."""
        # Ocultar mensaje inicial
        self.results_message.pack_forget()
        
        # Mostrar panel de resultados
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Limpiar lista de archivos
        self.eval_files_listbox.delete(0, tk.END)
        
        # Mostrar archivos evaluados
        for result in self.evaluation_results:
            self.eval_files_listbox.insert(tk.END, result['nombre'])
        
        # Configurar selección de archivos
        self.eval_files_listbox.bind('<<ListboxSelect>>', self.on_eval_file_selected)
        
        # Mostrar botones
        self.results_action_frame.pack(fill=tk.X, pady=10)
        
        # Mover a la siguiente pestaña
        self.notebook.tab(2, state="normal")
        self.notebook.select(3)
        
        # Detener indicador de progreso
        self.progress.stop()
        self.progress.pack_forget()
        self.status_bar.config(text="Generación de resúmenes completada")
        
        # Seleccionar primer resultado si hay alguno
        if self.evaluation_results:
            self.eval_files_listbox.select_set(0)
            self.on_eval_file_selected(None)
    
    def _finalize_summarization_error(self):
        """Finaliza la generación de resúmenes con error."""
        self.notebook.tab(2, state="normal")
        self.progress.stop()
        self.progress.pack_forget()
        self.status_bar.config(text="Error en la generación de resúmenes")
    
    def on_eval_file_selected(self, event):
        """Maneja la selección de un archivo evaluado."""
        # Obtener índice seleccionado
        selected_indices = self.eval_files_listbox.curselection()
        if not selected_indices:
            return
        
        idx = selected_indices[0]
        if idx >= len(self.evaluation_results):
            return
        
        # Obtener resultado
        result = self.evaluation_results[idx]
        
        # Actualizar interfaz
        self.result_title.config(text=f"Archivo: {result['nombre']}")
        self.best_model_label.config(text=result['mejor_modelo'])
        
        # Actualizar texto del resumen
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, result['mejor_resumen'])
        
        # Actualizar razonamiento
        self.reasoning_text.delete(1.0, tk.END)
        self.reasoning_text.insert(tk.END, result['evaluacion'])
    
    def open_results_folder(self):
        """Abre la carpeta de resultados."""
        output_folder = self.output_folder.get()
        if os.path.exists(output_folder):
            # Abrir carpeta de resultados según el sistema operativo
            if sys.platform == 'win32':
                os.startfile(output_folder)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', output_folder])
            else:  # Linux/Unix
                subprocess.run(['xdg-open', output_folder])
        else:
            messagebox.showwarning("Advertencia", "La carpeta de resultados no existe.")

def main():
    root = tk.Tk() if not CUSTOM_TK else ctk.CTk()
    app = BecasAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()