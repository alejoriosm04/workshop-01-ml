# 00 — Acuerdos metodológicos comunes

Esta es la fuente única de decisiones compartidas. Se fundamenta en
`workshop1.md`, en las diapositivas de preprocesamiento/metodología/regresión
logística y en los notebooks de KNN + Pipeline, regresión lineal y regresión
logística vistos en clase.

## Reglas no negociables

1. Usar `RANDOM_STATE = 42` en divisiones, validación cruzada y búsquedas.
2. Separar primero `X` e `y`; el target nunca entra al preprocesador.
3. Hacer split 70/15/15. Para clasificación, estratificar por `y`.
4. Ajustar cada `SimpleImputer`, escalador y encoder solo al entrenar el
   `Pipeline` sobre `X_train`.
5. Elegir modelo e hiperparámetros usando validación, nunca test.
6. En fase 7, unir train + validation, ejecutar CV sobre ese conjunto y evaluar
   una sola vez el modelo optimizado en el mismo test reservado.
7. Toda fila o columna excluida debe tener cantidad, razón y evidencia escrita.

## Decisiones de preprocesamiento

| Situación | Decisión común | Justificación de clase |
| --- | --- | --- |
| Numérica faltante | `SimpleImputer(strategy="median")` | La mediana es robusta ante outliers. |
| Categórica faltante | `SimpleImputer(strategy="most_frequent")` | La moda es la estrategia presentada para categorías. |
| Binaria | Indicador 0/1 mediante `OneHotEncoder(drop="if_binary")` o mapeo explícito documentado | En clase, las binarias se representan con cero y uno. |
| Nominal | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | Evita inventar un orden; el material lo recomienda para la mayoría de modelos. |
| Ordinal clínica | One-Hot para el modelo base, salvo justificación fuerte de distancias ordinales | KNN y regresión logística no deben asumir que los niveles tienen distancias iguales. |
| Escala numérica sin outliers relevantes | `StandardScaler` | Es el escalador usado en los notebooks de KNN, Ridge/Lasso y logística. |
| Escala numérica con outliers plausibles que se conservan | `RobustScaler` | La consigna lo permite; usa cuartiles, coherente con el criterio IQR. |

`sparse_output=False` se conserva para que KNN reciba una matriz densa, tal como
advierte el ejercicio de clase KNN + Pipeline.

## Plantilla de pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore", drop="if_binary", sparse_output=False
    )),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])
```

Cada notebook ajustará el escalador y las listas de columnas según su dataset.
No ejecutar `fit_transform` manualmente sobre validación o test.

## Función común de IQR

La función detecta; no elimina automáticamente. Los límites deben reportarse en
una tabla y su interpretación debe depender del dominio.

```python
def iqr_outlier_report(df, numeric_cols):
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        rows.append({
            "variable": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "limite_inferior": lower,
            "limite_superior": upper,
            "n_outliers": int(mask.sum()),
            "porcentaje_outliers": 100 * mask.mean(),
        })
    return pd.DataFrame(rows)
```

## División y evaluación

```python
# primera división: 70% train y 30% temporal
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=stratify_arg
)

# segunda división: 15% validation y 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE,
    stratify=y_temp if stratify_arg is not None else None
)
```

En regresión, `stratify_arg = None`. En clasificación, `stratify_arg = y`.

| Problema | Métricas requeridas | Criterio principal de selección |
| --- | --- | --- |
| Regresión | MAE, MSE, R² | MAE de validation, complementado con R² y brecha train/validation. |
| Clasificación | Accuracy, Precision, Recall, F1 y matriz de confusión | Recall de `Recurred=Yes`, sin ignorar Precision y F1. |

Tablas obligatorias para train/validation:

```text
Regresión:      Modelo | MAE_train | MSE_train | R2_train | MAE_val | MSE_val | R2_val
Clasificación:  Modelo | Accuracy_train | Precision_train | Recall_train | F1_train |
                       Accuracy_val   | Precision_val   | Recall_val   | F1_val
```

Para test se repiten las métricas por modelo en una tabla separada. Las matrices
de confusión van aparte, con ejes y clase positiva identificados.

## Cross-validation y búsqueda

- Regresión: `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Clasificación: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- La búsqueda se ejecuta sobre el `Pipeline` completo para impedir fuga de datos.
- Ridge/Lasso: `alpha` mayor implica más regularización L2/L1.
- Logística: `C` menor implica más regularización; L2 es ridge y L1 es lasso.
- KNN: explorar `n_neighbors` y `p` (1 = Manhattan, 2 = Euclidiana), como en
  el notebook de clase. `k` bajo puede sobreajustar; `k` alto puede subajustar.

## Nota para fase 9

Las conclusiones conjuntas deben conectar los resultados con: sensibilidad de KNN
a distancia/escala, sesgo-varianza de `k`, regularización L1/L2, discrepancia
train-validation, validación cruzada y limitaciones de ambos datasets.

