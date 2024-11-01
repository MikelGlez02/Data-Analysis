# Proyecto Tratamiento de Datos 2024/25

Data Processing Project in Python

# Autores

Miguel González Martínez (100451423), Carlos de Quinto (100451547)

# Descripción del Problema: ¿En qué consiste el proyecto?

En el proyecto se basa en una comparación de las prestaciones obtenidas al utilizar distintas vectorizaciones de los documentos y al menos dos estrategias distintas de aprendizaje automático.

# Objetivos del Proyecto

Hemos realizado los siguientes pasos:
1. Hemos analizado las variables de entrada, visualizando la relación entre la variable de salida y algunas de las categorías en la variable categories, en la que explicaremos su
potencial relevancia en el problema.
2. Implementación de un pipeline para el preprocesado de los textos (NLTK, Gensim o SpaCy). Tenga en cuenta que para trabajar con transformers el texto se pasa sin preprocesar.
3. Representación vectorial de los documentos mediante tres procedimientos diferentes: TF-IDF, Word2Vec(es decir, la representaci´on de los documentos como promedio de los embeddings de las palabras que lo forman) o Embeddings contextuales calculados a partir de modelos basados en transformers (e.g., BERT, RoBERTa, etc).
4. Entrenamiento y evaluación de modelos de regresión utilizando al menos las dos estrategias siguientes de aprendizaje automático: Redes neuronales utilizando PyTorch para su implementación. Al menos otra técnica implementada en la librería Scikit-learn (e.g., K-NN, SVM, Random Forest, etc)
5. Comparación de lo obtenido en el paso 3 con el fine-tuning de un modelo preentrenado con Hugging Face. En este paso se pide utilizar un modelo de tipo transformer con una cabeza dedicada a la tarea de regresión.

Se deberá utilizar la información en la variable directions y/o desc para todos los pasos del proyecto, pudiéndose combinar la información de estas variables con alguno de los metadatos en las otras variables. Deberá utilizar métricas para la evaluación adecuadas para los problemas de regresión. Las prestaciones de los distintos métodos deben estimarse con alguna metodología de validación que también deberá explicar en la documentación. Deberá aportar una descripción de la metodología empleada y analizar las prestaciones obtenidas según las variables de entrada.
Tenga en cuenta que el objetivo es describir el trabajo realizado y hacer un análisis crítico de los resultados obtenidos. Apóyese para ello en gráficas u otras representaciones que considere oportunas. No es necesario describir los algoritmos utilizados, aunque sí deberá explicar cómo ha realizado el ajuste de sus parámetros.

El trabajo de extensión es libre, deberá ampliar el proyecto básico en la dirección que considere más oportuna. Por ejemplo:
• Uso de un summarizer preentrenado (utilizando pipelines de Hugging Face) para proporcionar un resumen de la variable directions, la cual es una lista de instrucciones que puede contener textos relativamente grandes
así como pasos repetidos.
• Estudiar la capacidad de los modelos de tipo transformer para la generación de nuevas recetas. Aquí también se pueden comparar las prestaciones de esto respecto a su implementación con técnicas de prompting sobre modelos del lenguaje de uso libre (LLAMa, Mixtral, etc.).
• Explorar el potencial de técnicas de NLP como el uso de bigramas, part-ofspeech tagging, tesauros, etc., (explotando, por ejemplo, la funcionalidad disponible en la librería NLTK de Python).
• Comparación de prestaciones utilizando distintos embeddings contextuales, y visualización y análisis empleando técnicas basadas en grafos.

Tome esta lista como una mera sugerencia, puede elegir cualquier otro tema siempre que encaje dentro del ´ambito de la asignatura. En el trabajo de extensi´on se valorar´a la creatividad y originalidad en la elección. Si tiene dudas sobre la idoneidad de la extensión elegida, consulte con el profesor.

# Metodología aplicada

Procesamiento de los datos, modelos utilizados,

# Análisis de Resultados

A

# Bibliografía

A



