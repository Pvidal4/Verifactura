# Verifactura: Automatización de facturas vehiculares

## 📑Tabla de contenido

1.[📂 Descripción del problema](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=1.%20%F0%9F%93%82-,Descripci%C3%B3n%20del%20problema)

2. [⚙️ Dataset](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=2.%20%E2%9A%99%EF%B8%8F-,Dataset)
3.  [🤖 Metodología](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=3.%20%F0%9F%A4%96-,Metodolog%C3%ADa)
4. [📊 Resultados](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=4.%20%F0%9F%93%8A-,Resultados)
5. [🔑Instalación y uso](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=5.%F0%9F%94%91-,Instalaci%C3%B3n%20y%20uso)
6. [💻 Interfaz de usuario](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=6.%20%F0%9F%92%BB-,Interfaz%20de%20usuario)
7. [🔩 Estructura del proyecto](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=7.%20%F0%9F%94%A9-,Estructura%20del%20proyecto)
8. [⚖ Consideraciones éticas](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=8.%20%E2%9A%96-,Consideraciones%20%C3%A9ticas)
9. [🧑‍💻Autores y contribuciones](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=9.%20%F0%9F%A7%91%E2%80%8D%F0%9F%92%BB-,Autores%20y%20contribuciones)
10. [📜 Licencia](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=10.%20%F0%9F%93%9C-,Licencia)
11. [🤝 Agradecimientos y referencias](https://github.com/Pvidal4/Verifactura/blob/main/README.md#:~:text=11.%20%F0%9F%A4%9D-,Agradecimientos%20y%20referencias)

## 1. 📂 Descripción del problema

Las instituciones financieras enfrentan un cuello de botella en la validación de facturas vehiculares por la variedad de formatos y la dependencia de procesos manuales.
La digitación humana genera más del 15% de errores, causando reprocesos y demoras en los desembolsos. Esto incrementa el riesgo operativo y de fraude documental.
El desafío de Verifactura es automatizar la extracción y validación de datos clave de las facturas con alta precisión. Se busca alcanzar un 90% de exactitud y reducir en más del 80% el tiempo de procesamiento manual.

Es un plataforma creada para ser utilizada principalmente por la áreas operativas de las instituciones financieras previo al análisis crediticio de los clientes prospectantes.

## 2. ⚙️ Dataset

**Dataset inicial:** 26 facturas reales emitidas por diversas concesionarias ecuatorianas y almacenadas en la plataforma SCRIBD.

**Dataset de entrenamiento:** 500 registros sintéticos representativos de distintas combinaciones de marca, tipo, clase, capacidad, combustible, número de ruedas y valor total de la factura.

Cada registro fue clasificado en una de las cinco categorías de usuario: Familiar, Estudiante, Ejecutivo, Rural o Transporte público/comercial.

## 3. 🤖 Metodología
## 4. 📊 Resultados
## 5.🔑Instalación y uso
## 6. 💻 Interfaz de usuario
## 7. 🔩 Estructura del proyecto
El repositorio de Verifactura está constituido por las siguientes carpetas:

**app:** desarrollo de las aplicaciones utilizadas en Verifactura.

**docs:** dentro de esta carpeta se encuentra documentado el paso a paso del proyecto
* **planificacion:** planteamiento del problema, objetivos, cronograma, recursos y riesgos identificados en la etapa inicial.
* **análisis_datos:** detalle de la composición del Dataset_inicial, análisis exloratorio, estadística descriptiva, análisis bivariado, outliers, matriz de correlaciones.  
* **arquitectura:** 
* **optimización:** detalle de data de entrenamiento, definición de hiperparámetros, análisis de sensibilidad, partial dependece plots, ranking de hiperparámetros, análisis de interacciones. 
* **consideraciones éticas:** análisis de sesgo, riesgos identificados y medidas de mitigación, impacto social positivo y negativo, uso y mal uso de Verifactura, limitaciones.
* **manual de usuario:** guía paso a paso para usar la interfaz, capturas de pantallas anotadas, explicación de cada funcionalidad, troubleshooting (problemas comunes y soluciones), preguntas frecuentes (FAQ), información de contacto para soporte.

**imagenes:** carpeta con todos los gráficos y tablas obtenidas en los diferentes procesos de construcción de Verifactuta.

**models:**

**tests:**

**train:** arquitectura, requerimientos, modelo, README.

## 8. ⚖ Consideraciones éticas

### Aspectos éticos considerados

**Análisis de riesgo y medidas de mitigación**

* **Riesgo de equidad y fairness**

Se ha detectado un sesgo de género ocupacional, pues la implementación del sistema puede reproducir o amplificar desigualdades existentes al automatizar tareas predominantemente ocupadas por mujeres (digitación y verificación). 
**Estrategia:** Implementar monitoreo de sesgo de género en la automatización y establecer un programa de reconversión laboral y capacitación digital para los grupos más afectados (particularmente mujeres en roles de digitación).

* **Riesgo de Uso y resguardo de datos sensibles de facturas**

VeriFactura procesa facturas digitales que contienen datos personales y financieros de personas naturales (compradores de vehículos), pero los excluye del análisis, no así a la información comercial de las concesionarias. Si no se establecen controles estrictos, podría existir riesgo de re-identificación o acceso no autorizado a datos sensibles.
**Estrategia:** Adoptar un marco de gobernanza de datos personales que incluya encriptación, anonimización y controles de acceso basados en roles, conforme a la Ley Orgánica de Protección de Datos Personales (LOPDP).

* **Riesgo de Falta de explicabilidad y transparencia del modelo**

El proceso de extracción y validación automatizada se basa en modelos de IA que podrían no ser fácilmente interpretables por usuarios no técnicos. Esto puede limitar la comprensión de por qué una factura es aceptada o rechazada, reduciendo la confianza de los operadores humanos y de las concesionarias.

**Estrategia:** Incorporar herramientas de explicabilidad y trazabilidad de decisiones (por ejemplo, LIME o SHAP) y desarrollar una interfaz de usuario con reportes interpretables para las concesionarias y personal operativo.


### Limitaciones consideradas del modelo

Existe un alcance excluido, acerca de los escenarios que no cubre Verifactura:

* Procesamiento de Otros Documentos Bancarios
* Soporte para Idiomas Adicionales
* Aprobación Automática de Crédito
* Integración Directa con Sistemas Contables
  
### Uso y mal uso de Verifactura

**Uso dual:** Aunque fue diseñado para optimizar la gestión documental (extracción y validación) de facturas vehiculares, su arquitectura podría adaptarse para otros contextos en los que la extracción masiva de datos financieros o comerciales derive en vulneraciones éticas o legales. 

* Escenarios y medidas de prevención:

* Reutilización del modelo de extracción para procesar documentos personales o contractuales (nóminas, comprobantes de pago, escrituras, etc.) sin consentimiento explícito de los titulares.
* Integración con sistemas de vigilancia o scoring crediticio sin supervisión ética, lo cual podría derivar en prácticas discriminatorias o violaciones a la privacidad.
* Transferencia o entrenamiento secundario del modelo con datos no anonimizados, generando un riesgo de reidentificación o de sesgo no controlado.

**Mal uso:** se presenta cuando la herramienta se implementa, manipula o configura fuera de los controles previstos.

* Uso por personal no autorizado que acceda a datos sensibles o modifique parámetros del modelo sin registro en los logs.
* Elusión de protocolos de revisión humana, confiando plenamente en los resultados automáticos sin control de precisión o trazabilidad.
* Manipulación intencionada de facturas digitalizadas para obtener beneficios indebidos (por ejemplo, validaciones falsas o fraude documental).


## 9. 🧑‍💻 Autores y contribuciones

**Andrea Morán Vargas:** Científico de datos

**Pedro Vidal Orús:** AI Leader

## 10. 📜 Licencia
## 11. 🤝 Agradecimientos y referencias
