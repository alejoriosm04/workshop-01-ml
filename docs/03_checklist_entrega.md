# 03 — Checklist de entrega y revisión cruzada

Usen esta lista antes de subir el enlace del repositorio. Cada punto corresponde a
un requisito de `workshop1.md`.

## En cada notebook

- [ ] Se identifica el problema, el dataset, el target y el contexto de negocio/salud.
- [ ] Se muestran `head()`, `tail()`, `shape`, `info()`, `describe()` y diccionario de datos.
- [ ] Se interpretan al menos tres estadísticas numéricas en contexto.
- [ ] Se calcula número y porcentaje de nulos por columna y se justifica la estrategia.
- [ ] Se reportan duplicados antes/después de limpiarlos.
- [ ] Se revisan espacios, mayúsculas y etiquetas equivalentes en categorías.
- [ ] Se aplica la función IQR, se muestran límites/cantidades y se explica el tratamiento.
- [ ] Se incluyen barras, pie, histograma, scatter y boxplot con interpretación y recomendación.
- [ ] Se justifica encoding para cada grupo de variables categóricas.
- [ ] Se justifica el escalador según algoritmos y outliers.

## Pipeline y separación

- [ ] La división es 70/15/15 con `random_state=42`.
- [ ] Clasificación usa `stratify=y` en ambas divisiones.
- [ ] `Pipeline` contiene el `ColumnTransformer` y el modelo.
- [ ] Imputación, escalamiento y encoding se ajustan solo al entrenar con `X_train`.
- [ ] Hay una celda Markdown que explica por qué no existe fuga de datos.
- [ ] `OneHotEncoder` tiene `handle_unknown="ignore"` y salida densa para KNN.

## Modelos y métricas

- [ ] Uber entrena KNN Regressor, Ridge y Lasso.
- [ ] Thyroid entrena KNN Classifier, logística L2 y logística L1.
- [ ] Cada modelo tiene predicciones en train y validation.
- [ ] Las tablas train/validation contienen todas las métricas requeridas.
- [ ] Se discute overfitting/underfitting comparando train vs. validation.
- [ ] Thyroid incluye matrices de confusión y la discusión explícita de falso negativo/falso positivo.
- [ ] Se elige un modelo con trade-offs, no solo con el valor máximo de una métrica.

## Test, CV y muestra inventada

- [ ] Se evalúan los tres modelos en test antes de optimizar, como solicita la fase 6.
- [ ] Se compara el ranking validation vs. test sin usar test para cambiar decisiones.
- [ ] Se explica K-Fold con palabras propias.
- [ ] Se unen train + validation solamente después de la selección de fase 5.
- [ ] `RandomizedSearchCV` recibe el pipeline completo y la CV correcta.
- [ ] Se reportan `best_params_` y el desempeño final de test.
- [ ] La observación inventada usa exactamente las columnas esperadas por el modelo final.
- [ ] La predicción inventada se interpreta según el dominio y no se presenta como certeza.

## Cierre conjunto y repositorio

- [ ] Fase 9 compara KNN, regularización L1/L2, escala, sesgo-varianza y limitaciones.
- [ ] Cada notebook tiene un subplot 2×2 con sus gráficas más importantes.
- [ ] Se declaran las limitaciones específicas: calidad de coordenadas en Uber; tamaño, desbalance y temporalidad en Thyroid.
- [ ] Los notebooks corren desde cero y no contienen rutas personales ni errores.
- [ ] `data/raw/` no se sube al repositorio; `data/README.md` permite reproducir la descarga.
- [ ] El repositorio contiene los dos `.ipynb` ejecutados, `workshop1.md`, README y documentación.
- [ ] La plataforma recibe únicamente el enlace del repositorio, según la consigna.

