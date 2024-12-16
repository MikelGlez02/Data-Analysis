# Proyecto Final: Tratamiento de Datos 2024/25

## Autores
Miguel González Martínez (100451423) y Carlos de Quinto Cáceres (100451547)

# Análisis de Recetas y Predicción de Calificaciones con NLP y ML

Este repositorio contiene un proyecto que utiliza técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP) y modelos de aprendizaje automático para analizar y predecir calificaciones de recetas basadas en características textuales y datos numéricos. El conjunto de datos incluye recetas con atributos como ingredientes, categorías y calificaciones.

## Descripción del Proyecto

Los objetivos principales del proyecto son:
1. Preprocesar y extraer características significativas de los textos y metadatos de las recetas.
2. Experimentar con múltiples representaciones de texto (TF-IDF, Word2Vec y BERT) para la extracción de características.
3. Construir y evaluar modelos predictivos para las calificaciones de las recetas.


## Conjunto de Datos
El dataset proporcionado incluye 20,130 entradas con información sobre recetas (instrucciones, categorías, descripciones, entre otros). La tarea principal es predecir la variable `rating` usando otras variables textuales y numéricas como entrada.

## Características

- **Preprocesamiento de Texto**: Uso de `spaCy` para tokenización, lematización y eliminación de palabras vacías.
- **Ingeniería de Características**:
  - Vectorización con TF-IDF
  - Embeddings con Word2Vec
  - Embeddings contextuales con BERT
- **Modelos de Aprendizaje Automático**:
  - K-Nearest Neighbors (KNN)
  - Redes Neuronales (NN)
## Estructura de archivos del proyecto

Como se puede observar hay varios notebooks.
El notebook V4 es un intento fallido de implementear una cabeza de regresión en un modelo pre entrenado BERT para clasificación.
Los notebooks V3 son los notebooks que se han utilizado para los resultados finales.
Cada uno de ellos contiene los resultados con distintas features:
  - All data: Utiliza todos los datos de texto (instrucciones, categorias, descripciones, titulo) asi como numericos (calorías, grasas etc.)
  - OnlyTextdata: Utiliza solamente los datos de texto como la categoría anterior.
  - Only some Text Data: Utiliza solamente las columnas de instrucciones y descripciones.
  - Directions: Solo utiliza las instrucciones.
  - Descriptions: Solo utiliza las descripciones de las recetas.


## Workflow

En primer lugar se ha limpiado la base de datos de todos los valores NA que contenía, eliminando asi todas las recetas que contienen un NA tanto numerico como de texto.
Esto ha reducido el número de recetas a entorno a 10 000.
Es conveniente debido a que el tiempo de procesamiento de todo el dataset es muy elevado y esto permite la ejecución en local de este problema con gpu.

En segundo lugar se ha preprocesado el texto para las vectorizaciones que lo necesiten como TF-IDF.
Para el preprocesado del texto se ha usado spacy con el modelo 'en_core_web_sm'

Despues se procede a la vectorización de los datos con los 3 modelos

Por ultimo se entrenan los modelos y se evalúa su rendimiento

## Métricas utilizadas

## Resultados


### Visualización de Categorías de Recetas
- **Top 20 Categorías Más Valoradas**:
  ![Top Categorías](Top20Categorias.png)

- **Puntuaciones de Todas las Categorías**:
  ![Todas las Categorías](Categorias.png)



### Rendimiento de los Modelos



## Herramientas y Librerías para Posibles Mejoras Futuras del Proyecto

- **Procesamiento de texto:** NLTK, SpaCy, Transformers.
- **Aprendizaje Automático:** PyTorch, Scikit-learn.
- **Validación de Datos:** Pydantic.
- **Visualización:** Matplotlib.

## Bibliografía

### Documentación y herramientas utilizadas

1. **Python**: Lenguaje base del proyecto.
   - [Python Documentation](https://docs.python.org/3/)

2. **Jupyter Notebooks**: Para exploración interactiva.
   - [Jupyter Documentation](https://jupyter.org/documentation)

6. **PyTorch**: Framework de deep learning.
   - [PyTorch Documentation](https://pytorch.org/docs/)

7. **Scikit-learn**: Algoritmos clásicos de machine learning.
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

9. **Hugging Face Transformers**: Fine-tuning de modelos.
   - [Hugging Face Documentation](https://huggingface.co/docs/transformers/)

10. **Jupytext**: Conversión entre notebooks y scripts.
    - [Jupytext Documentation](https://jupytext.readthedocs.io/en/latest/)

