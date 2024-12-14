# Proyecto Final: Tratamiento de Datos 2024/25

## Autores
Miguel González Martínez (100451423) y Carlos de Quinto Cáceres (100451547)

## Introducción
Este proyecto forma parte del Máster en Ingeniería de Telecomunicación y tiene como objetivo aplicar técnicas avanzadas de procesamiento de datos y aprendizaje automático para resolver tareas relacionadas con documentos textuales. En este caso, se utilizará un conjunto de datos basado en recetas de cocina.

### Resumen del Proyecto

- **Procesado de datos textuales.**
- **Vectorización de documentos** con TF-IDF, Word2Vec y embeddings contextuales basados en Transformers.
- **Regresión** utilizando Redes Neuronales y técnicas clásicas de aprendizaje automático.

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

3. **Desarrollo con Python Avanzado:**
   - Uso de Pydantic para validación de datos.
   - Implementación de arquitecturas avanzadas como Redes Neuronales Convolucionales (CNN) y algoritmos de reducción de dimensionalidad como PCA.

## Esquema del Proyecto

```
          +------------------------------------------------+
          |        Análisis Exploratorio de Datos          |
          +------------------------------------------------+
                                |
          +------------------------------------------------+
          |        Preprocesamiento (NLKT, SpaCy)          |
          +------------------------------------------------+    
                                |
          +------------------------------------------------+
          |        Vectorización de Textos (TF-IDF)        |
          +------------------------------------------------+    
                                |
          +------------------------------------------------+
          |        Modelado (SVM,CART,NN,CNN,MSE,R2)       |
          +------------------------------------------------+    
                                |
          +------------------------------------------------+
          |                 Validación                     |
          +------------------------------------------------+
```

## Estructura del Proyecto

```plaintext
ProyectoTD/
|
├── Proyecto.ipynb            # Archivo .ipynb 
├── .gitignore                # Archivos y carpetas ignorados por Git
└── README.md                 # Documentación principal del proyecto
```

## Herramientas y Librerías para Posibles Mejoras Futuras del Proyecto

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

