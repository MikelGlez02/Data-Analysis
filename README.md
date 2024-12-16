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

## Resultados

### Visualización de Categorías de Recetas
- **Top 20 Categorías Más Valoradas**:
  ![Top Categorías](Top20Categorias.png)

- **Puntuaciones de Todas las Categorías**:
  ![Todas las Categorías](Categorias.png)

### Rendimiento de los Modelos
- **TF-IDF + KNN**:
  - Error Absoluto Medio (MAE): 0.828
  - Error Cuadrático Medio (MSE): 1.562
- **Word2Vec + Red Neuronal**:
  - MAE: 0.757
  - MSE: 1.529
- **BERT + Red Neuronal**:
  - MAE: 0.735
  - MSE: 1.415


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

