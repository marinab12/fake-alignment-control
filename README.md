# 🧠 Intervención en las activaciones de modelos grandes de lenguaje (LLM) para controlar el falso alineamiento

## 📌 Introducción

La **Inteligencia Artificial Generativa**, está experimentando un desarrollo acelerado que la posiciona como una herramienta clave en la innovación tecnológica. Su capacidad para generar contenido original abre nuevas oportunidades para la investigación y el desarrollo, al tiempo que exige replantear los marcos teóricos y prácticos tradicionales.  

En este contexto surge uno de los retos más relevantes: el **falso alineamiento**. Este fenómeno describe situaciones en las que los **modelos grandes de lenguaje (LLM)** aparentan cumplir con normas éticas y de seguridad cuando son supervisados, pero las eluden en ausencia de supervisión. Investigaciones recientes han mostrado que, si no se aborda adecuadamente, este comportamiento representa un desafío significativo para la **confiabilidad y seguridad** de estas tecnologías.  

Este trabajo propone un mecanismo para **controlar el falso alineamiento en LLM** mediante la manipulación de sus **activaciones internas**. Partiendo de un modelo pre-entrenado **LLaMa2-Chat 7B**, se ajusta para inducir dos comportamientos, uno alineado y otro que finge alineamiento. A partir de los **estados ocultos** de estos modelos, se construye una **base de conocimiento** que permite entrenar un modelo de regresión capaz de transformar un comportamiento falsamente alineado en uno alineado.  

Gracias a los patrones aprendidos por este regresor, se establece un criterio para **corregir el comportamiento** de los modelos que aparentan alineamiento, aportando un avance hacia sistemas más seguros y confiables.  


## 🚀 Prepara el entorno

Este proyecto utiliza un archivo `requirements.txt` para manejar las librerías necesarias.

### ✅ Si ya tienes un entorno de Python creado
1. Activa tu entorno:
   - **venv / virtualenv (Linux/Mac):**
     ```bash
     source ruta_del_entorno/bin/activate
     ```
   - **venv / virtualenv (Windows):**
     ```bash
     ruta_del_entorno\Scripts\activate
     ```
   - **conda:**
     ```bash
     conda activate mi_entorno_existente
     ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt

### 🆕 Si NO tienes un entorno de Python
1. Crea el entorno con venv:
   ```bash
   python -m venv fake-alignment-control_env
   ```
2. Activa tu entorno:
   - **Linux/Mac:**
     ```bash
     source fake-alignment-control_env/bin/activate
     ```
   - **Windows:**
     ```bash
     fake-alignment-control_env\Scripts\activate
     ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt


## 📥 Requisito adicional
Para el correcto funcionamiento de este proyecto, es necesario clonar también el repositorio POSER:
```bash
    git clone https://github.com/sevdeawesome/POSER.git
```