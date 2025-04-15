# Modelo de clasificación con Árbol de Decisión

## Descripción
Este proyecto implementa un modelo de **clasificación** basado en un **árbol de decisión** para predecir si las ventas de un conjunto de datos (`carseats.csv`) son altas o bajas. El programa incluye preprocesamiento de datos, codificación de variables categóricas, entrenamiento del modelo, evaluación de su rendimiento y optimización mediante **post-poda** (pruning) utilizando validación cruzada.

El objetivo es clasificar las ventas en dos categorías:
- **0**: Ventas altas (Sales > 8).
- **1**: Ventas bajas (Sales ≤ 8).

## Estructura del Programa
El programa sigue estos pasos principales:

1. **Carga de Datos**:
   - Se carga el archivo `carseats.csv` usando `pandas`.
   - Se crea una nueva variable binaria `ventas_altas` basada en la columna `Sales` (> 8 → 0, ≤ 8 → 1).
   - La columna original `Sales` se elimina.

2. **División de Datos**:
   - Los datos se dividen en conjuntos de entrenamiento (`X_train`, `y_train`) y prueba (`X_test`, `y_test`) usando `train_test_split` con una semilla aleatoria (`random_state=123`).

3. **Preprocesamiento**:
   - Las variables categóricas se codifican mediante **One-Hot Encoding** usando `ColumnTransformer` y `OneHotEncoder`.
   - Las variables numéricas se mantienen sin cambios (`remainder='passthrough'`).
   - Los datos preprocesados se convierten en DataFrames con nombres de columnas adecuados.

4. **Modelo Inicial**:
   - Se entrena un árbol de decisión (`DecisionTreeClassifier`) con una profundidad máxima de 5 y criterio `gini`.
   - Se visualiza la estructura del árbol y se calcula su **accuracy** en el conjunto de prueba.
   - Se genera una matriz de confusión para evaluar el rendimiento.

5. **Optimización con Post-Poda**:
   - Se realiza una búsqueda por validación cruzada (`GridSearchCV`) para encontrar el mejor valor del parámetro `ccp_alpha`, que controla la complejidad del árbol.
   - Se entrena un modelo final con el mejor `ccp_alpha` encontrado.
   - Se evalúa el modelo final y se calcula su **accuracy**.

6. **Importancia de Predictores**:
   - Se calcula y muestra la importancia de cada variable en el modelo final.

## Requisitos
Para ejecutar el programa, necesitas instalar las dependencias listadas en el archivo `requirements.txt`.

### Instalación de Dependencias
1. Asegúrate de tener Python 3.x instalado.
2. Instala las dependencias ejecutando:
   ```bash
   pip install -r requirements.txt