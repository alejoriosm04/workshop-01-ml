# 02 — Guía de Persona B: clasificación de recurrencia tiroidea

Fuente: <https://www.kaggle.com/datasets/jainaru/thyroid-disease-data/data>.

## Definición correcta del problema

Aunque la consigna lo llama diagnóstico tiroideo, este CSV no predice un diagnóstico
inicial. Predice si un paciente con cáncer tiroideo diferenciado tendrá **recurrencia**
después del tratamiento inicial. El target es `Recurred`:

```python
y = df["Recurred"].map({"No": 0, "Yes": 1})
```

La clase positiva es `Yes` (recurrencia). Un falso negativo es un paciente que sí
recurrirá, pero se clasifica como no recurrente; en un escenario de seguimiento,
podría perder vigilancia clínica oportuna. Un falso positivo implica seguimiento o
pruebas adicionales innecesarias. Por ello se prioriza Recall, sin aceptar una
Precision tan baja que el sistema genere demasiadas alertas inútiles.

## Perfil verificado del CSV original

Archivo: `data/raw/thyroid/Thyroid_Diff.csv`.

| Hallazgo | Resultado observado | Decisión |
| --- | ---: | --- |
| Filas y columnas | 383 × 17 | Dataset pequeño; usar `stratify` y CV estratificada. |
| Target | `Recurred` | `No`: 275; `Yes`: 108 antes de deduplicar. |
| Duplicados exactos | 19 | Eliminar, como exige la consigna; quedan 364 filas. |
| Clases tras deduplicar | `No`: 256; `Yes`: 108 | La recurrencia representa 29,7 %: hay desbalance moderado. |
| Nulos | 0 en las 17 columnas | Mostrar la tabla de nulos y conservar `SimpleImputer` dentro del pipeline para reproducibilidad. |
| Formato categórico | No se detectaron espacios ni diferencias de mayúsculas equivalentes | Ejecutar y reportar normalización preventiva con `.str.strip()`; no inventar cambios. |
| Numérica cruda | Solo `Age` | No forzar IQR sobre categorías nominales. |

## Fase 1: decisiones y justificaciones

### Fuga temporal: columna `Response`

`Response` describe la respuesta al tratamiento. Para un modelo que estima riesgo
de recurrencia desde el diagnóstico o la planificación del tratamiento inicial, esa
información puede estar disponible después del momento de predicción y producir fuga
temporal. Por coherencia, el notebook principal debe excluirla de `X` y declarar el
momento de uso del modelo:

```python
features_to_exclude = ["Recurred", "Response"]
```

Esta no es una eliminación arbitraria: protege la validez temporal de la predicción.
Si se incluyera `Response`, el objetivo cambiaría a pronóstico posterior a la
evaluación de respuesta, y habría que declararlo como otro caso de uso.

### Tipos y encoding

Después de excluir target y `Response` quedan:

```python
numeric_cols = ["Age"]
categorical_cols = [
    "Gender", "Smoking", "Hx Smoking", "Hx Radiothreapy",
    "Thyroid Function", "Physical Examination", "Adenopathy", "Pathology",
    "Focality", "Risk", "T", "N", "M", "Stage",
]
```

Usar `StandardScaler` para `Age`: no tiene outliers IQR y es el escalador principal
en los ejemplos de clase. Usar One-Hot para las categóricas, incluyendo `Risk`, `T`,
`N`, `M` y `Stage`. Aunque estas últimas tienen orden clínico, One-Hot evita asumir
que la distancia entre niveles es uniforme, algo importante para KNN y logística.

`OneHotEncoder(drop="if_binary")` convierte las binarias a un solo indicador 0/1
y codifica las restantes sin imponer orden falso.

### Outliers y visualizaciones

`Age` tiene Q1=29, Q3=51, IQR=22 y límites [-4, 84]. El rango observado es 15–82,
por lo que no hay outliers según IQR.

La consigna pide dos variables numéricas, pero el CSV solo contiene una variable
numérica cruda. Para cumplir sin aplicar IQR de forma incorrecta a una categoría
nominal, crear una variable exploratoria:

```python
risk_rank = {"Low": 1, "Intermediate": 2, "High": 3}
df_eda["Risk_rank"] = df_eda["Risk"].map(risk_rank)
```

`Risk` posee un orden clínico explícito; `Risk_rank` se usa únicamente para el
reporte IQR y el scatter de EDA. No se usa en el pipeline de modelado, donde `Risk`
permanece One-Hot. Si IQR marca una categoría alta como atípica, no se elimina: es
un caso clínicamente relevante, no un error de medición.

Gráficas mínimas:

- Barras y pie de `Recurred` para mostrar el desbalance.
- Histograma y boxplot de `Age`, coloreado o separado por recurrencia.
- Scatter `Age` vs. `Risk_rank`, coloreado por `Recurred`.
- Barras apiladas de `Risk`, `Response` (solo EDA) o `Stage` contra recurrencia.
- Barras de una variable binaria relevante, por ejemplo `Adenopathy` o `Smoking`.

En cada interpretación, distinguir asociación observada de causalidad clínica. Este
notebook es educativo y no constituye una herramienta médica de decisión.

## Fases 2 a 6: pipeline, modelos y evaluación

Usar el pipeline común con `StandardScaler` para `Age`. Los tres modelos iniciales:

```python
KNeighborsClassifier(n_neighbors=5, p=2)
LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5_000)
LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5_000)
```

`class_weight="balanced"` está justificado por el desbalance moderado y por la
prioridad del caso de recurrencia; la clase menciona pesos como alternativa al
desbalance. Explicar que puede aumentar Recall a costa de Precision.

Para train y validation calcular siempre:

```python
accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
```

Con `pos_label=1` y `zero_division=0` si se calculan métricas manualmente. No
seleccionar por Accuracy: un modelo que prediga siempre `No` ya tendría una Accuracy
aparentemente aceptable, pero Recall de recurrencia igual a cero.

En fase 6, evaluar los tres pipelines en test sin cambiar ningún parámetro a partir
de ese resultado.

## Fase 7: búsqueda de hiperparámetros

Unir train + validation y buscar solo sobre el modelo elegido en fase 5:

| Si gana | Distribución razonable |
| --- | --- |
| KNN | `model__n_neighbors`: impares 3–31; `model__p`: [1, 2]; `model__weights`: ["uniform", "distance"] |
| Logística L2 | `model__C`: logarítmica entre `1e-4` y `1e3`; `model__class_weight`: [None, "balanced"] |
| Logística L1 | `model__C`: logarítmica entre `1e-4` y `1e3`; `model__class_weight`: [None, "balanced"] |

Usar `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`,
`RandomizedSearchCV`, `scoring="recall"`, `n_iter=12`, `n_jobs=-1` y el pipeline
completo. Reportar los mejores parámetros, el recall promedio de CV y el resultado
final de test con todas las métricas, no solo Recall.

## Fase 8: observación inventada

Construir una sola fila con valores reales observados en el entrenamiento, por
ejemplo edad, antecedentes, función tiroidea, riesgo, T/N/M y estadio coherentes.
No incluir `Response`, pues el modelo principal no la usa. Reportar `predict()` y
`predict_proba()`; interpretar la salida como estimación educativa de recurrencia,
no diagnóstico clínico.

