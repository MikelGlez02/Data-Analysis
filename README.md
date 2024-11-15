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

Tome esta lista como una mera sugerencia, puede elegir cualquier otro tema siempre que encaje dentro del ´ambito de la asignatura. En el trabajo de extensi´on se valorar´a la creatividad y originalidad en la elección. Si tiene dudas sobre la idoneidad de la extensión elegida, consulte con el profesor.

# Metodología aplicada

Procesamiento de los datos, modelos utilizados,

# Análisis de Resultados

A

# Bibliografía


User Manual

1. Usage:
Run the application with

python run_labeler.py [--project_path PROJECT_PATH] [--url URL] [--user USER] [--tm TM]

Options are:

--project_path: The path to the labeling project folder. If it is not specified, the application will ask for it.
--url: A single url to be labeled. This option can be used to revise urls that have been wrongly labeled in a previous labeling session.
--user: Name identifying the labelling user. To use this option, you must select track_user: yes in the configuration file.
--tm: Transfer mode. Specifies the criterium used to import new data from the input folder. Available options are:
expand : All urls existing in the input folder are integrated into the dataset. This is the default option.
project : New URLs cannot be added to the dataset, but only information about labels or predictions.
contract: Only urls in the input folder are preserved in the data structure.
If the project folder does not exist, the application will create a new one, adding a copy of the default configuration file, and the execution stops so that the user can edit the configuration file (if applicable) and add some input file with urls to label.

Load the database.
Data import.
Data integration.
Selection of labels.
Labeling
Close.

2. Folder Structure
The application is integrated in the following folder structure

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
These are the minimal files required to run de labelling application. Besides of them, the application contains two data managmment tools,

run_db2file.py
run_file2db.py
run_pkl2json.py
and, also, a tool for analyzing the labeled dataset

run_analizer.py
common/lib/__init__.py
          /dataanalyzer/__init__.py
                       /ROCanalyzer.py

2. Databases and data files.
The input data and the results of the labelling application are usually stored in a mongo database or in a set of files.

2.1. Complete project file struture
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


3. Configuration file:
The configuration file is

config.cf
and must be located at the project folder. If it does not exist, the application creates one by copying a default configuration file (namely, file config.cf.default from the aplication folder structure). This file must be edited and adapted to the current labeling task.

This file contains several field, whose contents can be modified. The are described in the following secctions.

3.1. [DataPaths] Rutas de ficheros de datos
In this field the names of all the subfolders and files of the data folder can be modified

# Labelling subfolders
labeling_folder: labeling
input_folder: input
output_folder: output
used_folder: used

# Filenames
dataset_fname: dataset.pkl
labelhistory_fname: labelhistory.pkl
labels_endname: _labels
preds_endname: _predict.pkl
3.2. [Labeler] Categories
In this field, you define the categories you want to label, and the relationships between them. The set of categories must satisfy that, for each pair of categories, either they are disjoint to each other or one of them is a subcategory of the other. This allows the labeling of urls on category trees.

The set of categories is defined below. For example:

# List of categories. Every pair of categories A and B must satisfy 
# A in B, B in A, or intersection(A, B) = emptyset
categories: ['mammals', 'birds', 'primates']
The relationship between categories is specified by a dictionary in which the entry A:B means that A is a subclass of B

# Dictionary of dependencies between categories
# {A:B, C:D, E:D } means that A is a subclass of B and C and E are 
# subclasses of D
parentcat: {'primates': 'mammals'}
An additional field allows specifying if the categories of the first level are exhaustive or not

# If the categories are complete (i.e. they fill the observation space) then set
# fill_with_Other to no. Otherwise, set fill_with_Other to yes, and a category
# 'Other' will be added to the category set
fill_with_Other: yes
If they are not, 'yes' must be indicated. In this case, a new category 'Other' will be created in order to label those urls that do not fit into any of the first level categories.

With this definition, the following hierarchy of categories is established:

- 'mammals'
    - 'primates'
    - 'NO-primates'
- 'birds'
- 'Other'
Note that in order for the set of categories or subcategories in each level of the tree to be exhaustive, a "NO-category" is added to each level. However, the system will NOT add a button to the filler subcategories. In the previous example, the application will generate four labeling buttons:

- 'mammals'
- 'primates'
- 'birds'
- 'Other'
it being understood that those categories labeled as 'mammals' (and therefore, not labeled as 'primates') will be understood from the subcategory 'NO-primates'.

Therefore, during the execution of the application, the labeling window will show a button for each of the categories specified in the variable categories, together with (possibly) the category 'Other'. Additionally, a button will appear to label urls as ERROR in the case in which any problem with the access to the page prevents the labeling.

The following configuration parameters specify the values of the 4 possible labels that each category can have. These are the values that will appear in the output files.

# List of labels
yes_label: 1
no_label: -1
unknown_label = 0
error_label = -99
If any label with a different value from the previous ones is found in an input file, it will be replaced by an unknown_label tag by the application.

Finally, the application allows keeping a record of the user who has created each tagging event.

# Set the following to True if a user identifies will be requested on order 
# to track different labelers.
track_user: yes
3.3. [ActiveLearning] Parameters of the active learning algorithm.
# In multiclass cases, the reference class is the class used by the active
# learning algorithm to compute the sample scores.
ref_class: birds

# Max. no. of urls to be labeled at each labeling step
num_urls: 10

# Type of active learning algorithms
type_al: tourney

# AL threshold. This is used by the AL algorithm to compute the score
# of each  sample. A standard choice is to set it equal to the 
# decision threshold of the classifier model, but there may be good
# reasons for other choices. For instance, is classes are umbalanced, 
# we can take self.alth = 1 so as to promote sampling from the 
# positive class
alth: 1

# Probability of AL sampling. Samples are selected using AL with 
# probability p,
# and using random sampling otherwise.
p_al: 0.2

# Probability of selecting a sample already labelled
p_relabel: 0.2

# Size of each tourney (only for 'tourney' AL algorithm)
tourneysize: 40























A



