# sklearn_methods

## Descripción

Este repositorio contiene una colección de los métodos más utilizados en la librería scikit-learn, organizados por temáticas para facilitar su uso y adaptación en diversos proyectos de machine learning.

## Clasificación General y Explicación de los Grupos

### 1. Preprocesamiento de Datos

- **StandardScaler:** Escala características para tener media 0 y desviación estándar 1.
- **MinMaxScaler:** Escala características al rango [0, 1].
- **RobustScaler:** Escala características utilizando estadísticas que son robustas a valores atípicos.
- **Normalizer:** Normaliza muestras individualmente para tener norma unitaria.
- **Binarizer:** Binariza datos (convirtiéndolos a 0 o 1) basándose en un umbral.
- **PolynomialFeatures:** Genera nuevas características consistentes en todas las combinaciones de características hasta un grado dado.

### 2. Selección de Características

- **SelectKBest:** Selecciona las K mejores características según una función de puntuación.
- **RFE (Recursive Feature Elimination):** Selección recursiva de características para seleccionar un subconjunto de características.
- **PCA (Principal Component Analysis):** Reducción de dimensionalidad utilizando análisis de componentes principales.
- **VarianceThreshold:** Selecciona características según un umbral de varianza.

### 3. Modelos de Clasificación

- **LogisticRegression:** Regresión logística para clasificación binaria o multiclase.
- **KNeighborsClassifier:** Clasificador basado en los vecinos más cercanos.
- **SVC (Support Vector Classification):** Clasificador de máquinas de vectores de soporte.
- **DecisionTreeClassifier:** Árbol de decisión para tareas de clasificación.
- **RandomForestClassifier:** Bosque aleatorio para clasificación.
- **GradientBoostingClassifier:** Clasificador basado en boosting de gradiente.

### 4. Modelos de Regresión

- **LinearRegression:** Regresión lineal.
- **Ridge:** Regresión lineal con regularización L2.
- **Lasso:** Regresión lineal con regularización L1.
- **ElasticNet:** Combina Lasso y Ridge.
- **SVR (Support Vector Regression):** Regresión de máquinas de vectores de soporte.
- **RandomForestRegressor:** Bosque aleatorio para regresión.
- **GradientBoostingRegressor:** Regresor basado en boosting de gradiente.

### 5. Modelos de Clustering

- **KMeans:** Clustering basado en K-means.
- **AgglomerativeClustering:** Clustering aglomerativo jerárquico.
- **DBSCAN:** Clustering basado en densidad.
- **MeanShift:** Clustering basado en la estimación de la densidad.

### 6. Modelos de Ensamble

- **BaggingClassifier:** Clasificador basado en bagging.
- **BaggingRegressor:** Regresor basado en bagging.
- **AdaBoostClassifier:** Clasificador basado en boosting adaptativo.
- **AdaBoostRegressor:** Regresor basado en boosting adaptativo.
- **VotingClassifier:** Clasificador que combina múltiples clasificadores mediante votación.
- **VotingRegressor:** Regresor que combina múltiples regresores mediante promedio.

### 7. Validación y Evaluación

- **train_test_split:** Divide los datos en conjuntos de entrenamiento y prueba.
- **cross_val_score:** Evaluación cruzada para estimar el rendimiento del modelo.
- **GridSearchCV:** Búsqueda en cuadrícula para la optimización de hiperparámetros.
- **RandomizedSearchCV:** Búsqueda aleatoria para la optimización de hiperparámetros.
- **accuracy_score:** Calcula la precisión del modelo.
- **confusion_matrix:** Genera la matriz de confusión para la clasificación.

### 8. Métricas de Evaluación

- **precision_score:** Calcula la precisión de las predicciones.
- **recall_score:** Calcula el recall de las predicciones.
- **f1_score:** Calcula el F1-score.
- **mean_squared_error:** Calcula el error cuadrático medio.
- **r2_score:** Calcula el coeficiente de determinación (R²).

## Cómo Contribuir

Las contribuciones son bienvenidas. Si deseas contribuir, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-rama`).
3. Realiza tus cambios y confirma tus commits (`git commit -m 'Añadir nueva funcionalidad'`).
4. Sube tus cambios a tu fork (`git push origin feature/nueva-rama`).
5. Abre un pull request en GitHub.
