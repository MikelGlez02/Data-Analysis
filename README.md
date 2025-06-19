# Proyecto Final: Tratamiento de Datos 2024/25 (Convocatoria Extraordinaria)

## Autor
Miguel González Martínez (100451423, Convocatoria Extraordinaria) y Carlos de Quinto Cáceres (100451547, Convocatoria Ordinaria)

## Análisis de Recetas y Predicción de Calificaciones con NLP y ML

Este repositorio consiste en la extensión del proyecto realizado anteriormente en la convocatoria ordinaria con el objetivo de, no solo revisar, comparar y mejorar el proyecto base planteado anteriormente, sino incluyendo además el apartado de extensión (que no estaba incluido en la convocatoria ordinaria). Concretamente, se utiliza técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP) y modelos de aprendizaje automático de un conjunto de datos que servirá para analizar y predecir calificaciones de recetas basadas en características textuales y datos numéricos, además de proporcionar un resumidor entrenado para mejorar la experiencia del usuario. Cabe resaltar que no solo se irán detallando las diferencias entre la versión original y esta nueva versión durante el resúmen, sino que además se proporcionará un apartado específico en el que se expliquen en detalle las diferencias entre ambas versiones.

## Objetivos

1. **Preprocesamiento de datos**: Limpieza y extracción de características significativas de los textos y metadatos de las recetas.
2. **Representación del texto**: Experimentación con múltiples técnicas como TF-IDF, Word2Vec y BERT.
3. **Construcción de modelos predictivos**: Implementación y evaluación de modelos para predecir la calificación de recetas.
4. **Comparación con el proyecto base**: Corrección de errores metodológicos y optimización de estrategias anteriores.
5. **Extensión del proyecto**: Implementación de nuevas técnicas y enfoques.
6. **Implementación de tecnologías adicionales**: Integración de MongoDB, Kubernetes (K8s), Docker, Kafka, Graphene y PowerBI para mejorar la escalabilidad, visualización y distribución de datos.
7. **Aplicación de técnicas avanzadas de ML y estadística**: Implementación de CART, SVM, PCA, K-means, Autoencoders, Convolutional Neural Networks (CNNs) y métodos estadísticos como Fisher Matrix, Kalman Filter, Chapman-Kolgomorov y procesos de Markov.

## Conjunto de Datos

El dataset contiene **20,130 entradas** en formato JSON con información detallada sobre recetas, incluyendo:
- Instrucciones
- Categorías
- Descripciones
- Información nutricional (calorías, grasas, etc.)
- Calificación (`rating`, variable objetivo)

## Características del Proyecto

### Preprocesamiento de Texto
- Tokenización, lematización y eliminación de palabras vacías usando `spaCy`.
- Creación de representaciones vectoriales del texto mediante distintas técnicas.

### Ingeniería de Características
- **TF-IDF** para ponderar términos relevantes.
- **Word2Vec** para representar palabras en un espacio semántico.
- **BERT embeddings** para análisis contextual avanzado.

### Modelos de Aprendizaje Automático
- **K-Nearest Neighbors (KNN)** para clasificación basada en similitud.
- **Redes Neuronales (NN)** para una predicción más robusta.
- **Árboles de decisión CART** para interpretabilidad.
- **Máquinas de soporte vectorial (SVM)** para clasificación de alta dimensionalidad.
- **Análisis de Componentes Principales (PCA)** para reducción de dimensionalidad.
- **K-means** para agrupamiento de recetas similares.
- **Autoencoders** para aprendizaje de representaciones latentes.
- **Redes Neuronales Convolucionales (CNNs)** para clasificación avanzada.

### Extensiones Implementadas
- **Resumen de instrucciones**: Uso de `Hugging Face` para generar resúmenes de los pasos de preparación, y el uso de un resumidor entrenado.
- **Generación de recetas** en tiempo real con modelos `Transformer` y comparación con `LLAMA` y `Mixtral`.
- **Uso de técnicas avanzadas de NLP**: Análisis de bigramas, etiquetado gramatical (POS tagging), tesauros, etc.
- **Comparación de embeddings contextuales** para identificar el más adecuado.

## Estructura del Proyecto

Con respecto a la estructura del proyecto, habíamos implementado una serie de archivos .ipynb, con una V4 de un intento fallido de implementar una cabeza de regresión con un modelo pre entrenado BERT para clasificación, y una serie de archivos V3 para dar con los resultados finales, teniendo en cuenta los diferentes conjuntos de datos:
  - AllData: Utiliza todos los datos de texto (instrucciones, categorias, descripciones, titulo) asi como numericos (calorías, grasas etc.)
  - OnlyTextdata: Utiliza solamente los datos de texto como la categoría anterior.
  - OnlySomeTextData: Utiliza solamente las columnas de instrucciones y descripciones.
  - Directions: Solo utiliza las instrucciones.
  - Descriptions: Solo utiliza las descripciones de las recetas.

A diferencia del proyecto base, que contenía múltiples archivos `Jupyter Notebook` con código redundante, esta versión sigue una estructura modular en Python, dividiendo las funcionalidades en diferentes archivos `.py` con argumentos configurables:

```
proyecto_nlp_ml/
│── data/		# Conjunto de datos
│── src/		# Código fuente
│   ├── preprocess.py  # Preprocesamiento de texto
│   ├── feature_engineering.py  # Ingeniería de características
│   ├── model.py  # Entrenamiento y evaluación de modelos
│   ├── summarizer.py  # Entrenamiento y evaluación de modelos
│── notebooks/	# Notebooks para análisis exploratorio
│── results/	# Resultados de experimentos y modelos entrenados
│── deployment/	# Configuración de Docker y Kubernetes
│── main.py	# Script principal con parámetros configurables
│── requirements.txt  # Dependencias del proyecto
│── README.md	# Documentación del proyecto
```

## Ejecución del Proyecto

Se ejecuta el script principal con los argumentos correspondientes:
```bash
python main.py --mode train --data AllData --model KNN --summarizer PEGASUS
```

Parámetros disponibles:
- `--mode`: Define si se ejecuta en `train` (entrenamiento) o `evaluate` (evaluación).
- `--data`: Selecciona qué conjunto de datos usar (`AllData`, `OnlyTextData`, `Directions`).
- `--model`: Especifica el modelo a entrenar (`KNN`, `NN`, `BERT`, `SVM`, `CART`).
- `--summarizer`: Especifica el resumidor preentrenado del trabajo de extensión con Hugging Face (`BART`, `T5`, `BERTSUM`, `PEGASUS`).

De esta forma, se proporciona un modelo más dinámico de tal forma que, en vez de ejecutar con varios archivos de diferente código, simplemente se proporcionan 2 que permitan ejecutar varias posibilidades según lo que se considere.

## Convocatoria Ordinaria: Resultados

Se detallan los aspectos importantes:
- **Métricas utilizadas**: En nuestro proyecto, hemos usado la MAE para identificar el rendimiento del regresor, ya que denota cuanto se desvía, en promedio, la magnitud de los errores entre los valores que se han predecido, y los valores reales. En nuestro caso, nos salió de forma general un valor numérico de 0.828418629242508, que indica que, en promedio, las predicciones del regresor tienen un error absoluto de 0.828418629242508 respecto a los valores reales.
- **Limpieza de datos**: Se ha limpiado la base de datos de todos los valores NA que contenía, eliminando asi todas las recetas que contienen un NA tanto numerico como de texto, reduciendo el número de recetas a 10. Esto es conveniente a primeras debido a que el tiempo de procesamiento de todo el fichero de datos JSON es muy elevado, lo que permite la ejecución en local de este problema con la GPU.
- **Preprocesado**: Se ha preprocesado el texto para las vectorizaciones que lo necesiten como TF-IDF, mientras que para el preprocesado del texto se ha usado spacy con el modelo 'en_core_web_sm'
- **Vectorizado**: Despues se procede a la vectorización de los datos con los 3 modelos (TF-IDF, W2V, BERT con max_length=64 para evitar que colapse la memoria del ordenador)
- **Modelos utilizados**: Se han utilizado varios modelos para medir el rendimiento de los vectorizadores, como KNN de scikit learn, una red neuronal simple (SimpleNN) completamente conectada con una capa oculta de 128 neuronas y activación ReLU, seguida de una capa de salida con 1 neurona, es adecuada para tareas simples de regresión o clasificación binaria, y su estructura permite procesar entradas de tamaño definido por input_size. Por otro lado, ComplexNN es una red neuronal profunda y configurable con múltiples capas ocultas (por defecto 256, 128 y 64 neuronas) donde cada capa incluye activación ReLU, normalización BatchNorm, y Dropout para regularización, y es adecuada para tareas más complejas de regresión o clasificación, permitiendo personalizar tanto el tamaño de las capas ocultas como la tasa de dropout. Por último, también se proporciona RED-BERT pre entrenada de hugging face

### Visualización de Categorías de Recetas
- **Top 20 Categorías Más Valoradas**:
  ![Top Categorías](Top20Categorias.png)

- **Puntuaciones de Todas las Categorías**:
  ![Todas las Categorías](Categorias.png)

Es interesante observar que la mayoría de categorías tiene una media similar, que la desviación típica de los datos es menor que 1 y la mediana es muy estable. Podemos observar que exhibe un pico más alto y colas más pesadas en comparación con la distribución normal, lo que indica una distribución leptocurtica, lo que significa que los datos tienen una mayor concentración alrededor de la media y valores más extremos en las colas.

### Rendimiento de los Modelos y Conclusión

El rendimiento de los modelos se puede observar los excels. Hay un excel que recopila todos los datos(Resultado Datos), mientras que (Comparacion entre vectorizaciones) ayuda a comparar el rendimiento de las vectorizaciones dependiendo de los datos de entrada mencionados anteriormente. Aqui podemos ver los resultados para la red neuronal compleja:

| Model  | ALL DATA    | ONLY TEXT DATA | SomeTextData | Directions  | Descriptions |
|--------|-------------|----------------|-------------|-------------|--------------|
| W2V    | 0.729761541 | 0.72748363     | 0.70285362  | 0.72418654  | 0.7309196    |
| TF-IDF | 0.678939462 | 0.663268507    | 0.686691642 | 0.695839405 | 0.697468162  |
| BERT   | 0.711494803 | 0.644052863    | 0.690597415 | 0.720204115 | 0.697323203  |


Se pueden observar diversos patrones al cambiar de datos: la MAE del modelo bert pre entrenado con fine tunning resulta en una MAE de 2.82. Este resultado es la media -1, ya que el predictor siempre ha tenido como salida 1, y por tanto el MAE es la media -1. Esto sucede porque el modelo esta prediciendo siempre uno, lo que nos indica que la cabeza de regresion no se ha implementado correctamente. Los mejores resultados parecen ser obtenidos por BERT cuando se utilizan los datos adecuados. Es probable que un mejor rendimiento se logre aumentando el max length de BERT, siempre que se disponga de un equipo más potente. Por otro lado, ampliar aún más la red neuronal compleja podría también mejorar su rendimiento. En contraste, la red KNN resulta poco expresiva al manejar grandes volúmenes de datos.

## Convocatoria Extraordinaria: Extensión del Proyecto. ¿Qué diferencias hay?

Como se ha mencionado antes, con el objetivo de mejorar la experiencia del usuario al seguir las recetas con resúmenes más breves, he implementado un resumidor preentrenado mediante Hugging Face para prevenir así pasos repetitivos y redundantes, como respuestas extensas. Se van a utilizar 4 modelos preentrenados:
- **Bidirectional and Auto-Regressive Transformers (BART)**: Mantiene un equilibrio entre calidad gramatical y fidelidad, puede manejar entradas de longitud moderada.
- **Text-To-Text Transfer Transformer (T5)**: Un modelo que se ha entrenado en tareas NLP y es adaptable a múltiples tareas, incluyendo resumen.
- **Pre-training with Extracted Gap-sentences for Abstractive Summarization (PEGASUS)**: Entrenado con estrategia que simula el resumen real (gap-sentence), genera textos compactos con alta relevancia, y rinde excelente en benchmarks de resumen (como XSum, CNN/DailyMail).
- **Longformer Encoder-Decoder (LED)**: Diseñado para entradas muy largas (hasta 16,384 tokens o más), por lo que es ideal para datos extensos. Usa atención local y global eficiente, lo que le permite escalar sin saturar memoria.

Para realizar esto de forma práctica, simplemente se usa la función pipeline con el modelo definido en concreto que se debe utilizar, ya sea para BART (facebook/bart-large-cnn), T5 (t5-small), PEGASUS (google/pegasus-xsum) y LED (allenai/led-base-16384). Se aplicaron a las instrucciones de algunas recetas para así comprobar cómo de coherente y claras son las frases resumidas entre 75 y 100 palabras, por ejemplo. 

### Resultados

A continuación se presentan algunos resúmenes por cada modelo:








## Herramientas y Librerías Usadas del Proyecto

- **Procesamiento de texto:** NLTK, SpaCy, Transformers.
- **Aprendizaje Automático:** PyTorch, Scikit-learn. (Otra alternativa sería usar PySpark)
- **Validación de Datos:** Pydantic.
- **Visualización:** Matplotlib.

## Bibliografía

### Documentación y herramientas utilizadas

1. **Python**: Lenguaje base del proyecto.
   - [Python Documentation](https://docs.python.org/3/)

2. **Jupyter Notebooks**: Para exploración interactiva.
   - [Jupyter Documentation](https://jupyter.org/documentation)

3. **PyTorch**: Framework de deep learning.
   - [PyTorch Documentation](https://pytorch.org/docs/)

4. **Scikit-learn**: Algoritmos clásicos de machine learning.
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

5. **Hugging Face Transformers**: Fine-tuning de modelos.
   - [Hugging Face Documentation](https://huggingface.co/docs/transformers/)

6. **Jupytext**: Conversión entre notebooks y scripts.
    - [Jupytext Documentation](https://jupytext.readthedocs.io/en/latest/)

