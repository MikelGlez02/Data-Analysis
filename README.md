# Proyecto Final: Tratamiento de Datos 2024/25 (Miguel González Martínez, Carlos de Quinto Cáceres)

## Introducción
Este proyecto forma parte del Máster en Ingeniería de Telecomunicación y tiene como objetivo aplicar técnicas avanzadas de procesamiento de datos y aprendizaje automático para resolver tareas relacionadas con documentos textuales. En este caso, se utilizará un conjunto de datos basado en recetas de cocina.

### Resumen del Proyecto

- **Procesado de datos textuales.**
- **Vectorización de documentos** con TF-IDF, Word2Vec y embeddings contextuales basados en Transformers.
- **Regresión** utilizando Redes Neuronales y técnicas clásicas de aprendizaje automático.
- **Extensiones opcionales:** Generación de texto, análisis con grafos, visualización avanzada y uso de herramientas modernas como Docker, Kubernetes, y sistemas de Big Data.

## Conjunto de Datos
El dataset proporcionado incluye 20,130 entradas con información sobre recetas (instrucciones, categorías, descripciones, entre otros). La tarea principal es predecir la variable `rating` usando otras variables textuales y numéricas como entrada.

## Metodología

### Proyecto Básico
1. **Análisis Exploratorio de Datos:**
   - Visualización y análisis de la relación entre la variable de salida (`rating`) y las variables de entrada, incluyendo `categories`.

2. **Preprocesamiento:**
   - Normalización y limpieza de datos textuales con bibliotecas como NLTK, SpaCy o Gensim.
   - Tokenización y tratamiento especial para Transformers.

3. **Vectorización de Textos:**
   - Implementación de TF-IDF.
   - Uso de Word2Vec para promediar embeddings.
   - Generación de embeddings contextuales con modelos Transformers como BERT y RoBERTa.

4. **Modelado:**
   - **Redes Neuronales:** Implementadas con PyTorch para tareas de regresión.
   - **Técnicas adicionales:** Modelos como SVM, Random Forest o K-NN con scikit-learn.
   - **Fine-tuning:** Ajuste de modelos preentrenados utilizando Transformers de Hugging Face.

5. **Validación:**
   - Uso de técnicas como k-fold cross-validation y métricas adecuadas para evaluar el rendimiento.

### Extensiones
1. **Procesos avanzados de NLP:**
   - Uso de Summarizers para resumir instrucciones (`directions`).
   - Técnicas de generación de recetas con modelos de lenguaje como LLaMA o GPT-3.

2. **Análisis Avanzado:**
   - Visualización y análisis con herramientas de grafos.
   - Comparación de prestaciones entre distintos embeddings contextuales.

3. **Integración de Herramientas Modernas:**
   - **Docker y Kubernetes:** Contenerización y despliegue escalable de los modelos.
   - **Kafka y PySpark:** Procesamiento de datos en tiempo real.
   - **MongoDB:** Almacenamiento y gestión eficiente de datos no estructurados.
   - **Grafana y Power BI:**
     - Grafana para monitoreo en tiempo real de métricas del sistema y modelos.
     - Power BI para generación de dashboards interactivos y análisis visual avanzado.

4. **Desarrollo con Python Avanzado:**
   - Uso de Pydantic para validación de datos.
   - Implementación de arquitecturas de modelos avanzados como:
     - Redes Neuronales Autoencoder (Autoencoding NN).
     - Redes Neuronales Convolucionales (Convolutional NN).
     - Algoritmos CART, SVM, PCA.
     - Filtros de Kalman y Particle.

## Entrega
1. **Estructura del Repositorio GitHub:**
   - Código fuente bien organizado y documentado.
   - Notebooks o scripts para la preparación de datos y experimentación.
   - Resultados y análisis presentados en formato claro.

2. **Documentación:**
   - Archivo README.md con la descripción completa del proyecto.
   - Posibilidad de crear una página en GitHub Pages para explicar el trabajo.

3. **Fecha Límite:**
   - 13 de diciembre de 2024, 23:59 horas.

## Evaluación
1. **Proyecto Básico (2,25 puntos):**
   - Calidad de la metodología, documentación, y resultados obtenidos.

2. **Extensiones (0,75 puntos):**
   - Originalidad y calidad del trabajo adicional realizado.

## Instalación
1. Clonar este repositorio:
   ```bash
   git clone <repositorio>
   ```
2. Configurar un entorno virtual y las dependencias necesarias:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. Configuración de Docker y Kubernetes:
   - Asegúrese de tener Docker instalado y siga las instrucciones en el archivo `docker-compose.yml`.

4. Configuración de Kafka y PySpark:
   - Configure Kafka y PySpark utilizando los scripts en la carpeta `scripts/`.

5. Configuración de Grafana y Power BI:
   - **Grafana:** Siga la configuración en `grafana-config/` para importar dashboards predefinidos.
   - **Power BI:** Utilice los archivos `.pbix` en `powerbi-templates/` para configurar visualizaciones interactivas.

6. Ejecución del proyecto:
   ```bash
   python main.py
   ```

## Herramientas y Librerías
- **Procesamiento de texto:** NLTK, SpaCy, Gensim, Transformers.
- **Aprendizaje Automático:** PyTorch, Scikit-learn.
- **Big Data:** PySpark, Kafka, MongoDB.
- **Monitoreo y Visualización:** Grafana, Power BI.
- **Despliegue:** Docker, Kubernetes.
- **Validación de Datos:** Pydantic.
- **Visualización:** Matplotlib, Seaborn.

## Contribuciones
Las contribuciones son bienvenidas. Por favor, abra un issue o realice un pull request para discutir cualquier mejora.

## Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo LICENSE para más detalles.

