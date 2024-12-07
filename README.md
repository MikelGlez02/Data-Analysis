# Proyecto Tratamiento de Datos 2024/25

Data Processing Project in Python

# Autores

Miguel González Martínez (100451423), Carlos de Quinto (100451547)

# Descripción del Problema: ¿En qué consiste el proyecto?

En el proyecto se basa en una comparación de las prestaciones obtenidas al utilizar distintas vectorizaciones de los documentos y al menos dos estrategias distintas de aprendizaje automático.
A generic application for labelling a subset of sites in a web site collection, or a subset of docs in a text collection. I can be potentially used for classification and labelling problems over hierarchies of categories.
It is a labelling application that can be potentially used for classification and labelling problems over hiearchies of categories.

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

Tome esta lista como una mera sugerencia, puede elegir cualquier otro tema siempre que encaje dentro del ´ambito de la asignatura. En el trabajo de extensión se valorará la creatividad y originalidad en la elección. Si tiene dudas sobre la idoneidad de la extensión elegida, consulte con el profesor.

# Metodología aplicada

Procesamiento de los datos, modelos utilizados, etc

# Organización del Proyecto

Se ha organizado de la siguiente manera:

run_labeler.py
config.cf.default
common/__init__.py
common/lib/__init__.py
          /ConfigCfg.py
          /Log.py
          /labeler/__init__.py
                  /activelearning/__init__.py
                                 /activelearner.py
                  /labeling/__init__.py
                           /datamanager.py
                           /LabelGUIController.py
                           /labelprocessor.py
                           /LabelViewGeneric.py
                           /urlsampler.py
                           
These are the minimal files required to run de labelling application. Besides of them, the application contains two data managmment tools, and a tool for analyzing the labeled dataset

run_analizer.py
common/lib/__init__.py
          /dataanalyzer/__init__.py
                       /ROCanalyzer.py

The complete data structure for a labelling project is the following:

    project_path/.
                /dataset_labels.pkl     # Label dataset 
                /dataset_predicts.pkl   # Dataset of urls and predictions 
                /labelhistory.pkl   # Label event history
                /config.cf     # Configuration file
                /log           # Running event records
                /input/.
                      /urls.csv
                      /[cat1]_labels.csv  # File of new labels
                      /[cat1]_preds.pkl  # File of new predictions
                      /[cat2]_labels.csv  # File of new labels
                      /[cat2]_preds.pkl  # File of new predictions
                      ...
                /output/.
                       /labelhistory.csv  # Label history record
                       /[cat1]_labels.csv  # Label record about category cat1
                       /[cat2]_labels.csv  # Label record about category cat2
                       ...
                       /labelhistory.csv.old  # Old labelling events record
                       /[cat1]_labels.csv.old  # Old label record
                       /[cat2]_labels.csv.old  # Old label record
                       ...
                /used/.
                     /[cat1]_[labels|pred][codigo1].pkl
                     /[cat2]_[labels|pred][codigo2].pkl
                     /[cat3]_[labels|pred][codigo3].pkl
                     ...
The content of these files is explained in the following sections

2.2 Main working folder
We will call project_path to the path to the folder containing the labelling project (the name of this path can be specified in a configuration file, that is explained later). All data files related to the labelling process will be located in this folder.

2.3 Subfolders
The main project folder contains three subfolders:

input: It contains all input data
output: It contains all output files
used: It stores copies of all input files
(the name of these folders can be specified in the configuration file).


# Compilación/Ejecución del Proyecto

python main.py [--project_path PROJECT_PATH] [--url URL] [--user USER] [--tm TM]

--project_path: El arichivo del proyecto, el cual si no viene definido como tal, la aplicación lo preguntará.
--url: A single url to be labeled. This option can be used to revise urls that have been wrongly labeled in a previous labeling session.
--user: Nombre del usuario. Para usar esa opción, se usa track_user: en el archivo de configuración.
--tm: Modo de transferencia. Specifies the criterium used to import new data from the input folder. Available options are:
          expand : All urls existing in the input folder are integrated into the dataset. This is the default option.
          project : New URLs cannot be added to the dataset, but only information about labels or predictions.
          contract: Only urls in the input folder are preserved in the data structure.

# Análisis de Resultados

... The input data and the results of the labelling application are usually stored in a MongoDB or in a set of files.

# Bibliografía

Load the database.
Data import.
Data integration.
Selection of labels.
Labeling
Close.

# Archivo de Configuración 

The configuration file is config.cf and must be located at the project folder. If it does not exist, the application creates one by copying a default configuration file (namely, file config.cf.default from the aplication folder structure). This file must be edited and adapted to the current labeling task. This file contains several field, whose contents can be modified. The are described in the following sections.

