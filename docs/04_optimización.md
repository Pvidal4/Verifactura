# 1. Proceso de optimización de hiperparámetros
El presente análisis aborda la fase de sensibilidad e interpretación de hiperparámetros del modelo de clasificación Random Forest aplicado al sistema Verifactura, cuyo objetivo es predecir la categoría de usuario asociada a una factura en función de las características del vehículo y su costo total.

El dataset utilizado contiene 500 registros sintéticos representativos de distintas combinaciones de marca, tipo, clase, capacidad, combustible, número de ruedas y valor total de la factura.
Cada registro fue clasificado en una de las cinco categorías de usuario: Familiar, Estudiante, Ejecutivo, Rural o Transporte público/comercial.

Durante el proceso, se entrenó el modelo bajo tres espacios de búsqueda (A, B y C), con diferentes configuraciones de hiperparámetros. El espacio C, identificado como “configuración ligera o rápida”, presentó el mejor rendimiento con un score F1-macro de 0.99, evidenciando un equilibrio óptimo entre profundidad de los árboles, regularización y diversidad del bosque.

El ranking de importancia de los hiperparámetros indicó que el parámetro max_features (número de características consideradas por nodo) tiene el mayor impacto sobre el desempeño del modelo, seguido de max_depth, min_samples_split y max_samples. Esto sugiere que el control de la complejidad del modelo y la diversidad de los árboles son factores determinantes para optimizar la clasificación.

En síntesis, el modelo alcanzó un alto nivel de precisión y estabilidad, lo que permite avanzar hacia una fase de validación operativa e integración con el flujo real de facturación de la plataforma Verifactura.

# 2. Hiperparámetros explorados y rangos
Variables de entrada para el modelo de clasificación de Verifactura: marca, tipo, clase, capacidad, combustible, ruedas, total, con las mismas, podemos definir perfiles socioeconómicos o de uso vehicular que se encuentren asociados a estos valores. Una vez que se realiza la extracción y validación de las variables, el modelo debe aprender a predecir la categoría de usuario a la que pertenece una factura, basándose en las características del vehículo y su costo.

La categoría de usuario se encuentra definida por:

* Familiar: Vehículos medianos, entre 4 y 5 asientos, gasolina, marcas como Chevrolet, Hyundai, Toyota. Uso diario, desplazamiento urbano.
* Estudiante: Vehículos compactos, bajo costo total, cilindrada pequeña (≤1.4), gasolina. Movilidad personal, uso urbano.
* Ejecutivo: Vehículos sedán o SUV con marcas premium (Mazda, Nissan, Toyota, Kia alta gama) y totales altos (>25.000). Uso profesional, mayor confort.
* Rural: Vehículos 4x4, diésel, tipo camioneta o jeep, mayor tonelaje y ruedas reforzadas. Terrenos irregulares o zonas rurales.
* Transporte público / comercial: Capacidad ≥7, combustible diésel o gasolina, tipo bus, van o taxi, Transporte de pasajeros o carga.

El modelo de clasificación a utilizar es Random Forest y utiliza:
- N_estimators - Número de árboles
- max_depth — profundidad máxima de cada árbol
- max_features — número de características evaluadas en cada split
- min_samples_split — mínimo de muestras para dividir un nodo
- min_samples_leaf — mínimo de muestras en una hoja
- max_samples — proporción del dataset usada por cada árbol (submuestreo)

**Métricas iniciales**

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-1.png?raw=true)

**Hiperparámetros**

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-2.png?raw=true)

    
# 3. Resultados del análisis de sensibilidad

# 4. Partial dependence plots
Al comparar gráficamente los scores obtenidos por cada hiperparámetro, se visualiza en cada uno la tendencia al aumentar el score.

Max_depth, tiende a la baja cuando aumenta el score
![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-5.png?raw=true)

Max_features, comparte el mismo comportamiento a la baja

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-6.png?raw=true)

Se observa que el único con tendencia al alza es max_samples, que en el ranking ocupó el lugar número 4.

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-7.png?raw=true)



# 5. Ranking de importancia de hiperparámetros

Una vez ejecutado el modelo, se definió un ranking de importancia de correlación por score ordenando de mayor a menor cada Hiperparámetro
![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-4.png?raw=true)

Así, max_features o el número de características evaluadas es el más relevante para nuestro modelo, pues controla cuántos atributos se consideran para dividir en cada nodo, la lectura que tenemos es que al existir variables mixtas (numéricas y categóricas) tienen un alto impacto en el modelo.


# 6. Análisis de interacciones

Una vez ejecutadas la exploración sistémica en su totalidad, se obtuvo que el modelo óptimo era el Espacio de búsqueda: C, cuya característica principal era la configuración ligera o rápida (liviana), llegando a un Score de 0.99, con menor profundidad, robustez y regularización, pero conservando el log2.

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-8.png?raw=true)



# 7. Configuración final seleccionada y justificación

El ejercicio de sensibilidad permitió comprender de forma más profunda cómo los hiperparámetros afectan el desempeño del modelo Random Forest dentro del contexto del proyecto Verifactura.

Se comprobó que la diversidad de atributos evaluados (max_features) y la profundidad de los árboles (max_depth) son las variables con mayor influencia en la calidad de las predicciones. Estos parámetros controlan el equilibrio entre sesgo y varianza, evitando tanto el sobreajuste como la pérdida de información.

Asimismo, se observó que configuraciones más ligeras (con árboles menos profundos y muestreo parcial de datos (max_samples 0.7–0.8)) logran un desempeño sobresaliente con menor costo computacional. Esto valida la hipótesis de que un modelo más simple, pero bien regularizado, puede ser igual o más efectivo que uno complejo.

Desde una perspectiva práctica, este estudio ofrece una base sólida para la implementación del modelo en entornos reales, reforzando la importancia de mantener ciclos de monitoreo continuo, reentrenamiento controlado y documentación de versiones.

Finalmente, el trabajo demuestra que la comprensión del comportamiento interno del modelo es esencial no solo para optimizar métricas, sino también para garantizar transparencia, confiabilidad y equidad en los sistemas de inteligencia artificial aplicados a la gestión financiera y crediticia.

# 8. Comparación antes/después de la optimización

Una vez ejecutadas la exploración sistémica en su totalidad, se obtuvo que el modelo óptimo era el Espacio de búsqueda: C, cuya característica principal era la configuración ligera o rápida (liviana), llegando a un Score de 0.99, con menor profundidad, robustez y regularización, pero conservando el log2.

![alt text](https://github.com/Pvidal4/Verifactura/blob/main/imagenes/image-9.png?raw=true)





