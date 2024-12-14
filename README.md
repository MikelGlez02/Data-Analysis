# Proyecto Final: Tratamiento de Datos 2024/25

## Autores
Miguel González Martínez (100451423) y Carlos de Quinto Cáceres (100451547)

## Introducción
Este proyecto forma parte del Máster en Ingeniería de Telecomunicación y tiene como objetivo aplicar técnicas avanzadas de procesamiento de datos y aprendizaje automático para resolver tareas relacionadas con documentos textuales. En este caso, se utilizará un conjunto de datos basado en recetas de cocina.

### Resumen del Proyecto

- **Procesado de datos textuales.**
- **Vectorización de documentos** con TF-IDF, Word2Vec y embeddings contextuales basados en Transformers.
- **Regresión** utilizando Redes Neuronales y técnicas clásicas de aprendizaje automático.
- **Integración de herramientas modernas:** Docker, Kubernetes, Kafka, MongoDB y ELK Stack para visualización y monitoreo de logs.

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
   - Técnicas de generación de recetas con modelos de lenguaje como GPT.

2. **Análisis Avanzado:**
   - Visualización y análisis con herramientas de grafos.
   - Comparación de prestaciones entre distintos embeddings contextuales.

3. **Integración de Herramientas Modernas:**
   - **Docker y Kubernetes:** Contenerización y despliegue escalable de los modelos.
   - **Kafka:** Procesamiento de datos en tiempo real.
   - **MongoDB:** Almacenamiento y gestión eficiente de datos no estructurados.
   - **ELK Stack:** Centralización de logs para monitoreo y visualización avanzada.

4. **Desarrollo con Python Avanzado:**
   - Uso de Pydantic para validación de datos.
   - Implementación de arquitecturas avanzadas como Redes Neuronales Convolucionales (CNN) y algoritmos de reducción de dimensionalidad como PCA.

## Esquema del Proyecto

```
                            +-----------------------+
                            |   Kubernetes Cluster  |
                            +-----------------------+
                                      |
      +------------------+    +---------------+    +----------------+
      | Application Pod  |<-->|  Kafka Broker |<-->| MongoDB Service|
      +------------------+    +---------------+    +----------------+
              |                            |
        REST API                     Message Queue
```

## Estructura del Proyecto

```plaintext
ProyectoTD/
│
├── data/                     # Gestión de datos
│   ├── raw/                  # Datos originales (JSON proporcionado)
│   ├── processed/            # Datos procesados para modelos
│
├── kafka/                    # Configuración y scripts para Kafka
│   ├── producer.py           # Productor Kafka
│   ├── consumer.py           # Consumidor Kafka
│
├── k8s/                      # Configuración de Kubernetes
│   ├── deployment.yaml       # Despliegue de la aplicación
│   ├── kafka.yaml            # Despliegue de Kafka
│   ├── mongodb.yaml          # Despliegue de MongoDB
│   ├── service.yaml          # Exposición de la aplicación
│
├── src/                      # Código fuente principal
│   ├── __init__.py           # Inicializador del paquete principal
│   ├── main.py               # Punto de entrada principal
│   ├── preprocessing/        # Preprocesamiento de datos
│   │   ├── __init__.py        # Inicializador del subpaquete
│   │   ├── data_analysis.py   # Análisis avanzado de datos
│   │   ├── embeddings.py      # Generación de embeddings
│   │   ├── text_cleaner.py    # Limpieza y normalización de texto
│   │
│   ├── models/               # Modelado y evaluación
│   │   ├── __init__.py        # Inicializador del subpaquete
│   │   ├── regression.py      # Modelos de regresión
│   │   ├── transformers.py    # Fine-tuning con Transformers
│   │
│   ├── database/             # Integración con MongoDB
│   │   ├── __init__.py        # Inicializador del subpaquete
│   │   ├── mongodb_handler.py # Gestión de operaciones en MongoDB
│   │
│   ├── utils/                # Funcionalidades auxiliares
│       ├── __init__.py        # Inicializador del subpaquete
│       ├── arg_parser.py      # Parsing de argumentos de CLI
│       ├── logger.py          # Configuración de logs
│       ├── version_checker.py # Verificación de versiones de Python
│
├── tests/                    # Tests unitarios y de integración
│   ├── test_database.py       # Tests para base de datos
│   ├── test_models.py         # Tests para modelos
│   ├── test_preprocessing.py  # Tests para preprocesamiento
│
├── scripts/                  # Scripts de automatización
│   ├── entrypoint.sh          # Script para inicializar el entorno en contenedores
│
├── Dockerfile                # Contenedor Docker para la aplicación principal
├── docker-compose.yml        # Configuración para Kafka, MongoDB y ELK Stack
├── logstash.conf             # Configuración para Logstash (parte del ELK Stack)
├── requirements.txt          # Dependencias del proyecto
├── setup.py                  # Configuración para empaquetar como módulo de Python
├── .env                      # Variables de entorno
├── .gitignore                # Archivos y carpetas ignorados por Git
└── README.md                 # Documentación principal del proyecto
```

## Instalación

1. Clonar este repositorio:
   ```bash
   git clone <repositorio>
   cd proyecto_td
   ```

2. Levanta el entorno completo:
   ```bash
   docker-compose up --build
   ```

3. Para comprobar que los servicios funcionan bien:
   ```bash
   docker ps
   ```

4. Para ejecutar los diferentes pasos del proceso según los argumentos que pongamos
   - Preprocesamiento de datos:
     ```bash
     docker exec -it recipe_app python main.py preprocess --input_data data/raw/recipes.json --output_data data/processed/recipes_cleaned.json --preprocess_mode basic
     ```
   - Entrenamiento del modelo:
     ```bash
     docker exec -it recipe_app python main.py train --model_type pytorch --vectorizer tfidf --epochs 20 --batch_size 32 --learning_rate 0.001
     ```
   - Evaluación del modelo:
     ```bash
     docker exec -it recipe_app python main.py evaluate --model_type pytorch --evaluation_metric mae
     ```
   - Generar recetas en tiempo real:
     ```bash
     docker exec -it recipe_app python main.py generate_new_recipes
     ```
   - Ejecutar pruebas unitarias:
     ```bash
     docker exec -it recipe_app python main.py test
     ```

6. Acceder a Kibana para monitoreo:
   - Visita: `http://localhost:5601` y configura un índice con `logstash-*`.

## Herramientas y Librerías

- **Procesamiento de texto:** NLTK, SpaCy, Transformers.
- **Aprendizaje Automático:** PyTorch, Scikit-learn.
- **Big Data:** Kafka, MongoDB.
- **Monitoreo y Visualización:** ELK Stack (Elasticsearch, Logstash, Kibana).
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

