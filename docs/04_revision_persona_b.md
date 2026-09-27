# Revisión y cierre de Persona B

Fecha de revisión: 2026-09-26.

## Fases 1–2 revisadas

Se revisaron ambos notebooks contra `workshop1.md` y los acuerdos metodológicos:

- El target queda separado antes de construir los pipelines.
- Las divisiones son 70/15/15 con `random_state=42`; clasificación estratifica ambas divisiones.
- Imputación, escalamiento y One-Hot se ajustan exclusivamente sobre train.
- La evidencia de ausencia de fuga usa train y validation; test queda reservado para la fase 6.
- Uber usa `RobustScaler` por los trayectos largos plausibles y Thyroid usa `StandardScaler` para `Age`.
- La exclusión de `Response` en Thyroid queda justificada como prevención de fuga temporal.
- Las decisiones de limpieza, exclusión de columnas, outliers y encoding están documentadas.

Durante la revisión se corrigieron advertencias de compatibilidad de pandas/matplotlib,
se eliminaron rutas personales de los outputs y se corrigieron errores menores de redacción.

## Contrato B→C cerrado

| Problema | Modelo elegido | Evidencia principal | Modelo que debe optimizar C |
| --- | --- | --- | --- |
| Uber | Ridge (L2) | MAE val 2,2841; R² val 0,7716; brecha MAE 2,81 % | Ridge, búsqueda sobre `model__alpha` |
| Thyroid | Logística L2 | Recall val 0,7647; Precision 0,8125; F1 0,7879 | Logística L2, búsqueda sobre `model__C` y `model__class_weight` |

Los pipelines base y sus hiperparámetros están definidos en los notebooks. Persona C
debe preservar estas selecciones al iniciar las fases 6–8 y no usar test para cambiarlas.

## Fase 9 de clasificación

Persona B dejó compilados los aportes de datos y modelado, además del código del subplot
2×2. La conclusión de generalización solo puede cerrarse al integrar los resultados de
test, validación cruzada, `best_params_` y muestra inventada producidos por Persona C.
No se inventan ni anticipan esos resultados en esta rama.
