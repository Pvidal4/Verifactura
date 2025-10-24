# 1. Descripción detallada del Dataset
Para el proyecto *Verifactura* inicialmente utilizamos facturas vehiculares publicadas en la plataforma "Scribd", que es la biblioteca de ideas del mundo, con más de 200 millones de documentos. Permite encontrar y subir contenido sobre cualquier tema y nicho, desde artículos académicos y documentos legales hasta pasatiempos DIY, manuales y más.

Esta data inicial nos permitió trabajar con facturas reales de consecionarias ecuatorianas, y fueron etiquetadas en dos grupos según la información contenida: "COMPLETA" e "INCOMPLETA", el primera etiqueta corresponde a facturas reales de Venta de Vehiculos y la segunda a facturas de repuestos con datos en su mayoría incompletos.  

# 2. Estadísticas descriptivas
Realizando el resumen estadístico básico, sin ningún tratamiento de los datos, vemos que se está tomanto el RUC como una cantidad y no como una identificación que es lo correcto. Así mismo, se observa que las variables: SUBSIDIO, RUEDAS Y EJES, mantienen valores nulos en más del 60% de sus registros, por lo que no deben considerarse en el análisis, pues no existen sificientes datos para realizar una imputación de los mismos.
| RUBRO | SUBSIDIO | AÑO | SUBTOTAL | TOTAL | RUEDAS | DESCUENTO | EJES | IVA |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Count | 16.00 | 13.00 | 26.00 | 26.00 | 5.0 | 16.0 | 4.0 | 26.0 | 
| Media | 0.17 | 2018.07 | 11266.18 | 212703.33 | 4.0 | 211.44 | 2.0 | 1311.87  |
| Desv | 0.6675 | 7.7509 | 18564.88 | 20877.18 | 0.00 | 535.24 | 0.00 | 2239.96 |
| Min | 0.00 | 1998.00 | 0.88 | 0.99 | 4.00 | 0.00 | 2.00 | 0.00 |
| Max | 2.67 | 2025.00 | 81955.36 | 91790.00 | 4.00 | 1964.29 | 2.00 | 9834.64 |

**Identificación de variables categóricas y numéricas**

dtypes: float64(8), object(22)

**Estadísticas de tendencia central y dispersión- Análisis de asimetría y curtosis:**

| RUBRO | TOTAL | SUBTOTAL | IVA |
| :-----: | :-----: | :-----: | :-----: |
| MEDIA| 1203.33 | 11266.19 | 1311.87 |
| MEDIANA| 517.35 | 456.58 | 60.78 |
| MODA| 26 | 26 | 0 |
| ASIMETRÍA| 2.41 | 2.4431 | 2.5255 |
| CURTOSIS | 2.41 | 2.4431 | 2.5255 |

# 3. Visualizaciones del EDA

Al tener 8 variables numéricas, se espera que estas sean las que aporten con mayor información, sin embargo, existen valores faltantes en varias de ellas, por ello, se detalla el resumen de las tres variables numéricas más importantes: 

**TOTAL**

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-10.png?raw=true)

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-11.png?raw=true)

**SUBTOTAL**

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-12.png?raw=true)

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-13.png?raw=true)

**IVA**

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-14.png?raw=true)

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-15.png?raw=true)


# 4. Identificación de patrones, correlaciones, outliers

**Correlaciones**
Como se observa en el heatmap de correlaciones, las variables TOTAL e IVA, al ser variables dependientes del SUBTOTAL, tienen una correlación cercana a 1.

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-16.png?raw=true)

**Gráficos de Análisis bivariado**
*Top 3 Monto por marca*

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-17.png?raw=true)

*Top 3 Vehículos más comunes*

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-18.png?raw=true)

*IVA más alto por Tipo de Vehículo*

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-20.png?raw=true)

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-19.png?raw=true)

**Visualización de anomalías**

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-21.png?raw=true)

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-22.png?raw=true)

Como se observa el vehículo más nuevo contiene el Subtotal más Alto, sin embargo, el vehículo más antiguo no es el subtotal más Bajo, como podría esperarse en la interacción de las variables numéricas.

# 5. Decisiones de preprocesamiento justificadas

**Tratamiento de Outliers**

La segunda fase abordó el problema de los valores atípicos. Detectamos casos en los que el “TOTAL” de una factura duplicaba al “SUBTOTAL + IVA”, lo cual indicaba un error de extracción o digitación. Se aplicaron métodos estadísticos como el IQR (Interquartile Range) para marcar valores fuera de rango esperado y se implementaron transformaciones de winsorizing en los casos en los que no se podía justificar una eliminación. Con ello, buscamos preservar la mayor cantidad de información sin permitir que valores extremos sesgaran el modelo de clasificación.

**Estandarización de Formatos**

Finalmente, un problema recurrente fue la inconsistencia en formatos. Mientras algunas facturas expresaban el “CILINDRAJE” como “3900 C.C.”, otras lo hacían en valores enteros “2771” o incluso con separadores decimales europeos (“2.771,0”). Para superar esto, se diseñaron rutinas de normalización de texto y parsers numéricos que convirtieron todas las cifras a un formato estándar flotante en litros. De igual forma, las fechas se transformaron al formato ISO8601 (AAAA-MM-DD), lo que garantizó homogeneidad para posteriores análisis temporales.
En conjunto, este pipeline de limpieza permitió reducir en más de un 40% las inconsistencias detectadas durante el EDA y sentó las bases para un procesamiento más confiable en las siguientes etapas.

**Feature Engineering Avanzado**

Superada la limpieza, el paso siguiente fue generar un conjunto de características derivadas y transformadas que pudieran incrementar la capacidad predictiva de nuestros modelos. El feature engineering es especialmente relevante en proyectos como Verifactura, donde la información extraída de documentos no siempre se encuentra en un formato óptimo para algoritmos de clasificación o regresión.

*Creación de Variables Derivadas*

Se crearon nuevas variables a partir de combinaciones de campos existentes. Por ejemplo, se generó un indicador binario “vehículo_diésel” a partir del campo “COMBUSTIBLE”, útil para distinguir patrones de precios y clasificaciones de vehículos. También se introdujo la relación “precio_por_cilindrada” como cociente entre el “TOTAL” y la variable “CILINDRAJE”, lo cual permitió normalizar los precios en función de la potencia del motor.

*Encoding de Variables Categóricas*

Las variables categóricas representaron un reto significativo. Inicialmente se probó con One-Hot Encoding, que funcionó bien para variables con cardinalidad baja (“CLASE”: CAMIONETA, AUTOMÓVIL, CAMIÓN). Sin embargo, para variables con cardinalidad alta como “MODELO”, esta estrategia generó una explosión dimensional. Se exploraron métodos alternativos como Target Encoding (sustituir la categoría por el promedio del valor objetivo dentro de esa clase) y Frequency Encoding, que asignó a cada categoría la frecuencia relativa en el dataset. Esto último resultó más eficiente en términos computacionales.

*Transformaciones de Variables Numéricas*

Las variables numéricas fueron normalizadas utilizando StandardScaler (Z-score), lo que permitió que atributos como “SUBTOTAL”, “IVA” y “TOTAL” tuvieran media 0 y desviación estándar 1. Esta normalización facilitó el trabajo de algoritmos sensibles a la escala. También se probó con Robust Scaling, lo cual fue útil frente a la presencia residual de outliers.

*Selección de Características*

Para evitar el sobreajuste, se realizó una etapa de feature selection mediante análisis de importancia de características en RandomForest y regularización LASSO. Descubrimos que atributos como “CLASE”, “MARCA” y “CILINDRAJE” aportaban más información que “COLOR” o “DIRECCIÓN”. Esto nos permitió reducir dimensionalidad sin perder capacidad explicativa.

*Extracción de Características de Dominio*

Dado que trabajamos con documentos de facturación vehicular, se exploraron técnicas específicas del dominio. Por ejemplo, en los textos libres de descripción de ítems se aplicó un bag-of-words con TF-IDF para capturar términos relevantes como “DOBLE CABINA” o “4X4”, que muchas veces definían la clase del vehículo mejor que los campos estructurados.


# 6. Manejo de datos faltantes o desbalanceados

La primera fase consistió en identificar variables con alto porcentaje de valores faltantes. Para algunas columnas, como “MODELO HOMOLOGADO ANT”, el nivel de missing superaba el 60%, lo que obligaba a replantear su valor analítico. En lugar de eliminar estas variables de forma directa, se optó por aplicar estrategias mixtas de imputación. Para variables numéricas (ejemplo: “CILINDRAJE” o “AÑO”), se evaluó la imputación por media y mediana, concluyendo que la mediana resultaba más robusta frente a valores atípicos. Para variables categóricas (ejemplo: “CLASE” o “COLOR”), la estrategia de moda resultó más apropiada. Sin embargo, también se exploró la imputación avanzada mediante algoritmos como KNN Imputer, lo que demostró que es posible predecir categorías ausentes en función de vecinos más cercanos, aunque a costa de mayor tiempo de cómputo.

Uno de los mayores desafíos identificados fue el desbalance de clases. Durante el EDA, observamos que el número de facturas para “AUTOMÓVIL” superaba significativamente al de “JEEP”, generando problemas para entrenar modelos justos. Sin un balance adecuado, los algoritmos tienden a favorecer la clase mayoritaria, produciendo métricas engañosas.
Para abordar esto se probaron múltiples técnicas. El undersampling permitió equilibrar el dataset reduciendo facturas de categorías sobre-representadas, aunque a costa de perder información. Por otro lado, el oversampling aleatorio duplicó registros de clases minoritarias, pero introdujo el riesgo de sobreajuste. El método más prometedor fue SMOTE (Synthetic Minority Oversampling Technique), que genera ejemplos sintéticos en el espacio de características, proporcionando un balance más natural. Complementariamente, se exploraron técnicas híbridas como SMOTEENN, que combinan oversampling con limpieza de ruido.
A través de estas técnicas, logramos reducir el desbalance original y observar mejoras en métricas como el F1-macro, aunque los resultados seguían limitados por el tamaño reducido del dataset inicial.

**Data Augmentation**

El data augmentation buscó incrementar el volumen y diversidad de los datos a partir de variaciones de los existentes. Aunque es más común en imágenes y texto, exploramos su aplicación en el dominio tabular.
En las facturas vehiculares, se probaron métodos como la inyección de ruido gaussiano en atributos numéricos (ejemplo: variaciones mínimas en SUBTOTAL) y técnicas como Mixup, que genera nuevos registros combinando pares de ejemplos. A nivel de texto, se implementaron ensayos de paraphrasing automático en descripciones de vehículos, lo que permitió diversificar los términos sin alterar el sentido.
El impacto de estas técnicas fue moderado, principalmente porque el dataset original era limitado y las variaciones generadas podían introducir ruido. Sin embargo, sentaron las bases para un enfoque más robusto cuando se disponga de un corpus mayor.

**Partición Estratificada de Datos**

La división del dataset en subconjuntos de entrenamiento, validación y prueba fue realizada con estratificación, de manera que la proporción de clases se mantuviera estable. Inicialmente, el dataset pequeño dificultó esta partición: al reservar 15% para test y validación, algunas clases quedaban con apenas 1 o 2 ejemplos. Esto generó errores en la implementación original (ejemplo: ValueError: The test_size = 2 should be greater or equal to the number of classes = 3).
Este hallazgo fue crucial: puso en evidencia la insuficiencia de datos para entrenamiento supervisado tradicional. A pesar de implementar soluciones parciales, como aumentar el tamaño de entrenamiento o aplicar validación cruzada estratificada, los resultados no fueron lo suficientemente robustos. La matriz de confusión mostró una confusión recurrente entre “CAMIONETA” y “CAMIÓN”, lo que reflejaba que el modelo no captaba adecuadamente las diferencias entre ambas categorías.

**Pipeline de Preprocessing Automatizado**

La culminación del proyecto consistió en diseñar un pipeline automatizado, inspirado en la estructura propuesta por scikit-learn, que integrara limpieza, generación de características, normalización y balanceo. El objetivo era lograr un sistema modular y reutilizable. El código se estructuró en componentes como:

"from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
preprocessing_pipeline = Pipeline([
    ('cleaner', DataCleaner()),
    ('feature_engineer', FeatureEngineer()),
    ('scaler', StandardScaler()),
    ('balancer', SMOTEBalancer())
])"

Este diseño teórico se implementó parcialmente en preprocessing_pipeline.py, pero los resultados empíricos confirmaron nuestras sospechas: con datasets pequeños y facturas heterogéneas, el enfoque puramente supervisado no alcanza. Esto justificó un cambio hacia el pipeline documental verifactura_pipeline, basado en extracción por reglas, normalización y validación, mucho más adecuado para el contexto de OCR y documentos legales.







