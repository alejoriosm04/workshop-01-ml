# Workshop 01 — Pipelines, regresión y clasificación

Este repositorio contiene dos notebooks autónomos, construidos a partir de
`workshop1.md`, el material de clase en `Machine_Learning_Applied` y los CSV
originales descargados de Kaggle.

| Notebook | Dataset | Target |
| --- | --- | --- |
| `notebooks/01_regresion_uber.ipynb` | Uber Ride Price Prediction | `fare_amount` |
| `notebooks/02_clasificacion_thyroid.ipynb` | Thyroid Disease Data | `Recurred` (`Yes` = 1) |

## Reparto del equipo (3 personas)

La división es por etapa técnica y cada persona aporta a **ambos notebooks**. Así
nadie es dueño exclusivo de un notebook y la tercera persona no queda limitada a
reunir conclusiones.

| Persona | Rol | Fases | Aporte en regresión | Aporte en clasificación |
| --- | --- | --- | --- | --- |
| Persona A | Datos y preprocesamiento | 1–2 | EDA, limpieza, split y pipeline de Uber | EDA, limpieza, split y pipeline de Thyroid |
| Persona B | Modelado base y diagnóstico | 3–5 | KNN/Ridge/Lasso, métricas train–val, selección | KNN/logística L1/L2, métricas train–val, selección |
| Persona C | Generalización y optimización | 6–8 | Test, K-Fold, búsqueda, muestra inventada | Test, K-Fold, búsqueda, muestra inventada |
| Las tres | Conclusiones | 9 | Aporte por especialidad + subplot 2×2 | Aporte por especialidad + subplot 2×2 |

Cada etapa entrega un contrato explícito a la siguiente, definido en
[Acuerdos metodológicos comunes](docs/00_acuerdos_metodologicos.md). Respetar esos
contratos permite trabajar en paralelo sin rehacer trabajo ajeno.

### Integración y revisión cruzada

| Responsabilidad | Persona |
| --- | --- |
| Compilar la fase 9 de regresión | Persona A |
| Compilar la fase 9 de clasificación | Persona B |
| Ejecutar ambos notebooks desde cero y cerrar el checklist | Persona C |
| Revisar las fases 1–2 de ambos notebooks | Persona B |
| Revisar las fases 3–5 de ambos notebooks | Persona C |
| Revisar las fases 6–8 de ambos notebooks | Persona A |

## Orden de trabajo

1. Lean y acuerden `docs/00_acuerdos_metodologicos.md` antes de modificar los notebooks.
2. Persona A publica splits, columnas y pipelines: es la ruta crítica.
3. B y C adelantan su código contra las interfaces acordadas mientras A termina.
4. Persona B anuncia el modelo ganador de cada problema; solo entonces C lanza las búsquedas.
5. La fase 9 se escribe a tres manos por especialidad; cada integrador la compila.
6. Cada revisor valida su etapa antes de la ejecución final desde cero.

La guía de trabajo está en:

- [Acuerdos metodológicos comunes](docs/00_acuerdos_metodologicos.md)
- [Guía de regresión: Uber](docs/01_regresion_uber.md)
- [Guía de clasificación: Thyroid](docs/02_clasificacion_thyroid.md)
- [Checklist de entrega](docs/03_checklist_entrega.md)
- [Instrucciones de datos](data/README.md)

Los datos locales están ignorados por Git para evitar subir los archivos descargados.
El repositorio sí debe incluir los dos notebooks ejecutados, las celdas Markdown de
justificación y esta documentación.
