# Taller 1: Pipelines, regresión y clasificación

Machine Learning Applied

**Integrantes:** Alejandro Rios, Lina Ballesteros y Danna Salazar

Este repositorio contiene la solución del Taller 1 (enunciado en [`workshop1.md`](workshop1.md)). Se resuelven dos problemas de aprendizaje supervisado siguiendo el ciclo completo: EDA y limpieza, división train/validation/test, pipelines con `ColumnTransformer`, comparación de modelos, evaluación en test, validación cruzada con `RandomizedSearchCV`, predicción sobre una muestra inventada y conclusiones. Cada decisión está justificada en celdas Markdown dentro de los notebooks.

## Notebooks

| Notebook | Problema | Dataset | Target |
| --- | --- | --- | --- |
| [`01_regresion_uber.ipynb`](notebooks/01_regresion_uber.ipynb) | Regresión | [Uber Ride Price Prediction](https://www.kaggle.com/datasets/kushsheth/uber-ride-price-prediction) | `fare_amount` (tarifa en USD) |
| [`02_clasificacion_thyroid.ipynb`](notebooks/02_clasificacion_thyroid.ipynb) | Clasificación | [Thyroid Disease Data](https://www.kaggle.com/datasets/jainaru/thyroid-disease-data/data) | `Recurred` (recurrencia del cáncer, `Yes` = 1) |

Cada notebook está organizado por las fases del enunciado (1 a 9) y se entrega ejecutado, con todas sus tablas y gráficas.

## Resumen de resultados

### Regresión: tarifa de viajes de Uber

- Limpieza principal: se excluyeron 4.410 filas (2,2 %) con tarifas no positivas, coordenadas fuera de Nueva York o combinaciones imposibles de tarifa y distancia. La distancia del viaje se calculó con la fórmula de Haversine y se derivaron hora, día, mes y año.
- Modelos comparados: KNN Regressor, Ridge y Lasso.
- Modelo seleccionado: Ridge. En test obtuvo un MAE de 2,22 USD y un R² de 0,79. KNN mostró sobreajuste (su MAE sube 26 % de train a validación).
- Optimización: con `RandomizedSearchCV` y 5-Fold se eligió alpha ≈ 658. La mejora fue marginal (MAE de 2,21 USD en test).

### Clasificación: recurrencia de cáncer tiroideo

- Limpieza principal: se eliminaron 19 duplicados exactos y se excluyó la variable `Response`, porque se conoce después del tratamiento y anticipa el resultado.
- Modelos comparados: KNN Classifier, regresión logística L2 y regresión logística L1.
- Métrica prioritaria: Recall, porque no detectar una recurrencia (falso negativo) es el error más costoso.
- Modelo seleccionado: regresión logística L2. En test detectó 15 de 16 recurrencias (Recall 0,94).
- Optimización: la búsqueda de C con `RandomizedSearchCV` y 5-Fold estratificado no mostró una mejora clara. Con C ≈ 0,013, el Recall en test fue 0,81 y la Precision 0,87. Con solo 55 pacientes en test, estas diferencias son orientativas.

## Estructura del repositorio

```text
.
├── README.md
├── workshop1.md                  # Enunciado del taller
├── requirements.txt
├── data/
│   └── README.md                 # Instrucciones para descargar los datos
└── notebooks/
    ├── 01_regresion_uber.ipynb
    └── 02_clasificacion_thyroid.ipynb
```

## Cómo ejecutar

1. Usar Python 3.11 o superior (scikit-learn 1.8 o superior lo requiere). Los notebooks se ejecutaron con Python 3.14 y scikit-learn 1.9.

2. Crear el entorno e instalar las dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # En Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Descargar los datos de Kaggle y ubicarlos así (ver [`data/README.md`](data/README.md)):

   ```text
   data/raw/uber/uber.csv
   data/raw/thyroid/Thyroid_Diff.csv
   ```

4. Abrir los notebooks con Jupyter o VS Code y ejecutar todas las celdas en orden. Los notebooks encuentran los datos tanto si se abren desde la raíz del repositorio como desde la carpeta `notebooks/`. El de Uber tarda unos minutos por la predicción de KNN.

Los datos no se incluyen en el repositorio. Toda la limpieza se hace en memoria dentro de los notebooks, sin modificar los archivos originales.
