{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
import matplotlib.pyplot as plt\
import seaborn as sns\
\
# Cargar el conjunto de datos Iris\
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"\
column_names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]\
iris_data = pd.read_csv(url, names=column_names)\
\
# Mostrar las primeras filas del conjunto de datos\
print("Primeras filas del conjunto de datos:")\
print(iris_data.head())\
\
# Resumen estad\'edstico\
print("\\nResumen estad\'edstico:")\
print(iris_data.describe())\
\
# Comprobar valores nulos\
print("\\nValores nulos en el conjunto de datos:")\
print(iris_data.isnull().sum())\
\
# Visualizaci\'f3n de distribuciones\
plt.figure(figsize=(12, 6))\
sns.histplot(iris_data['SepalLength'], bins=30, kde=True)\
plt.title('Distribuci\'f3n de la Longitud del S\'e9palo')\
plt.xlabel('Longitud del S\'e9palo (cm)')\
plt.ylabel('Frecuencia')\
plt.show()\
\
# Visualizaci\'f3n de relaciones entre caracter\'edsticas\
plt.figure(figsize=(10, 6))\
sns.pairplot(iris_data, hue='Species')\
plt.title('Matriz de Dispersi\'f3n del Conjunto de Datos Iris')\
plt.show()\
\
# Visualizaci\'f3n de correlaciones\
plt.figure(figsize=(10, 6))\
correlation_matrix = iris_data.corr()\
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')\
plt.title('Matriz de Correlaci\'f3n')\
plt.show()}