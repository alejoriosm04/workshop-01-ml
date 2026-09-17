# 01 — Guía de Persona A: regresión de tarifas Uber

Fuente: <https://www.kaggle.com/datasets/kushsheth/uber-ride-price-prediction>.

## Perfil verificado del CSV original

Archivo: `data/raw/uber/uber.csv`.

| Hallazgo | Resultado observado | Decisión |
| --- | ---: | --- |
| Filas y columnas | 200.000 × 9 | Trabajar sobre el CSV original, conservando trazabilidad de limpieza. |
| Target | `fare_amount` | Problema de regresión: tarifa continua en USD. |
| `Unnamed: 0` | 200.000 valores únicos | Excluir de `X`: índice heredado, no atributo del viaje. |
| `key` | Solo 3.600 valores únicos | Excluir de `X`: identificador truncado/no predictivo; no representa una característica del viaje. |
| Nulos | 1 en `dropoff_longitude` y 1 en `dropoff_latitude` | Quedan cubiertos por el imputador; revisar si pertenecen a la misma fila. |
| Duplicados exactos | 0 | Reportar explícitamente que no se eliminaron filas duplicadas. |
| `fare_amount <= 0` | 22 filas: 17 negativas y 5 en cero | Excluir solo estas filas: una tarifa negativa o cero no es un precio válido para el target. Reportar la cantidad. |
| `passenger_count` inválido | 709 ceros y 1 valor mayor que 6 | Convertir esos 710 valores a `NaN` y dejar que el imputador de mediana del pipeline los trate. |
| Coordenadas fuera del área de estudio | 4.371 filas (2,186 %) usando caja amplia NYC | Excluir con justificación tras visualizar el mapa: son errores de captura, no viajes largos plausibles. |

La caja geográfica propuesta es longitud `[-74.3, -73.6]` y latitud `[40.4, 41.0]`;
es suficientemente amplia para el área urbana y aeropuertos. Antes de aplicarla,
muestren el scatter de coordenadas y documenten que el grupo excluido queda fuera
de la nube geográfica principal.

## Fase 1: decisiones y justificaciones

### Target, tipos y feature engineering

1. Conservar `fare_amount` únicamente como `y`.
2. Convertir `pickup_datetime` con `pd.to_datetime(..., utc=True, errors="coerce")`.
   Reportar cuántas fechas no se pudieron convertir.
3. Crear `hour`, `day_of_week`, `month` y `year`. Tratarlas como categóricas de
   baja cardinalidad y codificarlas con One-Hot: sigue el criterio del ejercicio
   de clase, donde variables discretas de pocos niveles se tratan como categorías.
4. Crear `trip_distance_km` con la fórmula de Haversine a partir de coordenadas
   válidas. Esta es una característica de dominio: resume origen y destino en una
   distancia útil para estimar una tarifa.
5. Excluir las cuatro coordenadas crudas después de construir `trip_distance_km`.
   Mantenerlas duplicaría la misma señal geográfica y dificulta que modelos de
   distancia interpreten el trayecto. Esta exclusión debe explicarse, no hacerse
   silenciosamente.

Después de estas decisiones:

```python
numeric_cols = ["trip_distance_km"]
categorical_cols = ["passenger_count", "hour", "day_of_week", "month", "year"]
```

### Outliers: dos variables exigidas

Sobre los datos originales, el reporte IQR arroja:

| Variable | Q1 | Q3 | Límite superior | Outliers | Decisión |
| --- | ---: | ---: | ---: | ---: | --- |
| `fare_amount` | 6,00 | 12,50 | 22,25 | 17.167 (8,583 %) | Conservar tarifas positivas altas: pueden corresponder a trayectos largos. |
| `trip_distance_km`* | 1,256 | 3,906 | 7,880 | 16.156 (8,258 %) | Conservar: trayectos largos son plausibles. |

\*Distancia calculada tras filtrar coordenadas inválidas; su máximo observado fue
36,69 km. No eliminar por IQR: IQR marca rareza estadística, no error de captura.

Como la distancia conserva una proporción relevante de outliers plausibles, usar
`RobustScaler` para `trip_distance_km`. Esta es una excepción justificada al
escalador estándar de clase y está contemplada en la consigna. Las columnas One-Hot
no se escalan.

### Gráficas mínimas que cumplen la fase 1

- Histograma de `fare_amount`, antes y después de marcar los valores no positivos.
- Barras y pie de `passenger_count` válido.
- Scatter `trip_distance_km` vs. `fare_amount`, con línea de tendencia opcional.
- Boxplots de `fare_amount` y `trip_distance_km`.
- Scatter de `pickup_longitude`/`pickup_latitude` para sustentar el filtro
  geográfico.

Cada gráfica requiere un párrafo: hallazgo, impacto de negocio y decisión. Por
ejemplo, tarifas largas no se borran porque un servicio real debe atender también
viajes caros, no solo el caso promedio.

## Fases 2 a 6: pipeline, modelos y evaluación

Usar el preprocesador común, sustituyendo `StandardScaler()` por `RobustScaler()`
en el pipeline numérico. Ajustar el pipeline completo sobre `X_train` y generar
predicciones en train, validation y test.

Modelos base requeridos:

```python
KNeighborsRegressor(n_neighbors=5, p=2)
Ridge(alpha=1.0)
Lasso(alpha=0.1, max_iter=10_000)
```

Para cada modelo, registrar MAE, MSE y R² en train y validation. Seleccionar el
modelo por el menor MAE de validation solo si no tiene una brecha preocupante
frente a train. R² y tiempo de inferencia complementan la decisión.

En fase 6, evaluar los tres modelos una vez en test y comprobar si el orden de
validation se conserva. Una gran caída en test indica que la validación no fue
representativa, que hubo selección excesiva o que el modelo no generaliza bien.

## Fase 7: búsqueda de hiperparámetros

Unir train y validation. Aplicar `RandomizedSearchCV` al pipeline del mejor modelo:

| Si gana | Distribución razonable |
| --- | --- |
| KNN | `model__n_neighbors`: enteros impares 3–31; `model__p`: [1, 2]; `model__weights`: ["uniform", "distance"] |
| Ridge | `model__alpha`: distribución logarítmica entre `1e-4` y `1e3` |
| Lasso | `model__alpha`: distribución logarítmica entre `1e-4` y `10` |

Usar `KFold(n_splits=5, shuffle=True, random_state=42)`, `scoring="neg_mean_absolute_error"`,
`n_iter=12` y `n_jobs=-1`. La cantidad moderada de iteraciones es deliberada: el
dataset es grande y, en especial, la búsqueda de KNN puede tardar bastante.

## Fase 8: observación inventada

Crear una fila con las columnas ya ingenierizadas, por ejemplo un viaje de 3 km,
dos pasajeros, viernes a las 18:00, mes y año presentes en entrenamiento. No usar
categorías nunca vistas para que la predicción sea interpretable. Explicar si el
precio estimado concuerda con una hora de alta demanda y una distancia urbana media.

