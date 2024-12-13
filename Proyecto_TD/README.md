# Proyecto Final: Tratamiento de Datos 2024/25

## Autores
Miguel González Martínez (100451423) y Carlos de Quinto Cáceres (100451547)

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

## Estructura del Proyecto

```plaintext
ProyectoTD/
│
├── data/                     # Gestión de datos (puede ser opcional si todo se almacena en MongoDB)
│   ├── raw/                  # Datos originales (JSON proporcionado)
│   ├── processed/            # Datos procesados para modelos
│
├── kafka/                    # Configuración y scripts para Kafka
│   ├── producer.py           # Productor Kafka
│   ├── consumer.py           # Consumidor Kafka
│   ├── topics/               # Configuración de tópicos
│
├── k8s/                      # Configuración de Kubernetes
│   ├── deployment.yaml       # Despliegue de la aplicación
│   ├── kafka.yaml            # Despliegue de Kafka
│   ├── mongodb.yaml          # Despliegue de MongoDB
│   ├── service.yaml          # Exposición de la aplicación y MongoDB
│
├── src/                      # Código fuente principal
│   ├── __init__.py
│   ├── main.py               # Punto de entrada principal
│   ├── preprocessing/        # Preprocesamiento de datos
│   │   ├── __init__.py
│   │   ├── text_cleaner.py   # Limpieza y normalización de texto
│   │   ├── embeddings.py     # Generación de embeddings
│   │
│   ├── models/               # Modelado y evaluación
│   │   ├── __init__.py
│   │   ├── regression.py     # Modelos de regresión (SVM, PyTorch)
│   │   ├── transformers.py   # Fine-tuning con Transformers
│   │
│   ├── database/             # Integración con MongoDB
│   │   ├── __init__.py
│   │   ├── mongodb_handler.py # Gestión de operaciones en MongoDB
│   │
│   ├── utils/                # Utilidades
│   │   ├── __init__.py
│   │   ├── arg_parser.py     # Parsing de argumentos
│   │   ├── logger.py         # Configuración de logs
│   │   ├── version_checker.py # Comprobación de versiones de Python
│
├── tests/                    # Tests unitarios y de integración
│   ├── test_preprocessing.py # Tests para el preprocesamiento
│   ├── test_models.py        # Tests para los modelos
│   ├── test_database.py      # Tests para las interacciones con MongoDB
│
├── Dockerfile                # Contenedor Docker para la aplicación principal
├── docker-compose.yml        # Configuración para Kafka, MongoDB, y Zookeeper
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Documentación del proyecto
├── setup.py                  # Configuración para convertirlo en paquete Python
├── .env                      # Variables de entorno (con conexión a MongoDB)
└── .gitignore                # Archivos a excluir en Git
```


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
   - Se va a basar en diferentes ejecuciones según su implementación:
        **Preprocesamiento de datos**
        ```bash
        python main.py preprocess --input_data data/raw/recipes.json --output_data data/processed/recipes_cleaned.json --preprocess_mode basic
        ```
        **Entrenamiento del modelo utilizado**
        ```bash
        python main.py train --model_type pytorch --vectorizer tfidf --epochs 20 --batch_size 32 --learning_rate 0.001
        ```
        **Evaluación del modelo**
        ```bash
        python main.py evaluate --model_type pytorch --evaluation_metric mae
        ```
        **Generar de nuevas recetas**
        ```bash
        python main.py generate_new_recipes
        ```

## Herramientas y Librerías
- **Procesamiento de texto:** NLTK, SpaCy, Gensim, Transformers.
- **Aprendizaje Automático:** PyTorch, Scikit-learn.
- **Big Data:** PySpark, Kafka, MongoDB.
- **Monitoreo y Visualización:** Grafana, Power BI.
- **Despliegue:** Docker, Kubernetes.
- **Validación de Datos:** Pydantic.
- **Visualización:** Matplotlib, Seaborn.

## Bibliografía

### Documentación y herramientas utilizadas

1. **Python**: Lenguaje base del proyecto.
   - [Python Documentation](https://docs.python.org/3/)

2. **Jupyter Notebooks**: Para exploración interactiva.
   - [Jupyter Documentation](https://jupyter.org/documentation)

3. **MongoDB**: Base de datos para almacenamiento semiestructurado.
   - [MongoDB Documentation](https://www.mongodb.com/docs/)

4. **Kafka**: Mensajería en tiempo real.
   - [Apache Kafka Documentation](https://kafka.apache.org/documentation/)

5. **Kubernetes**: Orquestación de contenedores.
   - [Kubernetes Documentation](https://kubernetes.io/docs/)

6. **PyTorch**: Framework de deep learning.
   - [PyTorch Documentation](https://pytorch.org/docs/)

7. **Scikit-learn**: Algoritmos clásicos de machine learning.
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

8. **Docker**: Contenerización del proyecto.
   - [Docker Documentation](https://docs.docker.com/)

9. **Hugging Face Transformers**: Fine-tuning de modelos.
   - [Hugging Face Documentation](https://huggingface.co/docs/transformers/)

10. **jupytext**: Conversión entre notebooks y scripts.
    - [Jupytext Documentation](https://jupytext.readthedocs.io/en/latest/)

Esta bibliografía proporciona referencias clave sobre las herramientas utilizadas y sus documentaciones oficiales, ayudando a entender y extender las funcionalidades implementadas en este proyecto.

## Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo LICENSE para más detalles.

