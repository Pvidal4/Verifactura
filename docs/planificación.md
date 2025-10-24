# 1. Definición del problema y Objetivos
## Definición
Las instituciones financieras que gestionan créditos vehiculares enfrentan un cuello de botella operativo en la validación y registro de facturas provenientes de concesionarias, debido a la gran variabilidad de formatos, baja calidad de escaneos y alta dependencia de procesos manuales.
Actualmente, la digitación y verificación humana generan errores en más del 15% de los casos, ocasionando reprocesos, demoras en los desembolsos y un incremento del riesgo operativo y de fraude documental.
El reto principal consiste en automatizar la extracción, validación y estandarización de datos clave de las facturas (como totales, VIN, fechas, IVA, Marca, Tipo, Clase, Capacidad, Tipo de combustible, # de Ruedas) manteniendo una precisión aceptable (90%) y una reducción del tiempo de procesamiento superior al 80% respecto al método manual.

## Objetivo general
Diseñar e implementar un prototipo funcional de sistema de inteligencia artificial híbrido (visión + lenguaje) que automatice la lectura, validación y estructuración de facturas vehiculares en múltiples formatos, alcanzando al menos un 90% de precisión en la extracción de datos clave y una reducción del 80% en el tiempo promedio de procesamiento, en un período de 6 semanas.
## Objetivos específicos
1. Desarrollar y entrenar un modelo OCR avanzado capaz de procesar facturas en PDF e imagen con distintas resoluciones y tipografías, logrando un mínimo de 90% de exactitud en reconocimiento de texto durante las pruebas de validación.
Duración: Semanas 1–2

2. Implementar un modelo de lenguaje (LLM) de extracción y normalización que convierta los datos reconocidos por OCR en un formato estructurado JSON, verificando coherencia en campos críticos (totales, VIN, fechas, IVA, Marca, Tipo, Clase, Capacidad, Tipo de combustible, # de Ruedas).
Duración: Semanas 3–4

3. Integrar reglas de negocio y una interfaz de revisión humana que permita validar automáticamente las facturas con alta confianza y derivar a revisión las de baja probabilidad, con registro de retroalimentación para fine tuning.
Duración: Semanas 5–6

4. Identificar el perfil o categoría del usuario del vehículo utilizando un modelo de clasificación supervisado basado en sus principales variables, con el fin de enfocar de manera más efectiva las estrategias de venta y marketing.
Duración: Semanas 5–6

# 2. Justificación y relevancia del proyecto

El proyecto propuesto responde a una necesidad crítica dentro del sector financiero, particularmente en el ámbito de los créditos vehiculares, donde la gestión documental constituye un punto neurálgico para la eficiencia operativa y la mitigación del riesgo. Actualmente, las entidades financieras deben procesar un alto volumen de facturas con múltiples formatos, calidades de imagen y estructuras heterogéneas, lo que genera demoras, errores humanos y vulnerabilidad ante fraudes documentales. Estos problemas impactan directamente en los tiempos de desembolso, la satisfacción del cliente y el cumplimiento normativo.

La automatización inteligente del procesamiento de facturas mediante una arquitectura híbrida de inteligencia artificial (visión + lenguaje) representa una solución innovadora y estratégica. Esta integración permite leer, interpretar y validar información crítica de forma automática, garantizando coherencia entre los datos y minimizando la intervención manual. Al reducir el tiempo de procesamiento en más de un 80% y mantener una precisión superior al 90%, el sistema no solo incrementa la eficiencia operativa, sino que también disminuye los costos asociados a reprocesos y auditorías internas.

Adicionalmente, la incorporación de un modelo de clasificación supervisado que identifique el perfil o categoría del usuario del vehículo aporta valor comercial al habilitar una segmentación inteligente de clientes. Esto permite focalizar estrategias de venta y diseñar productos crediticios adaptados a distintos perfiles, incrementando así la efectividad comercial y la competitividad del negocio.
Desde una perspectiva institucional, el proyecto fortalece el cumplimiento regulatorio al asegurar trazabilidad y validación automatizada de documentos, y desde el punto de vista tecnológico, impulsa la madurez digital del sector financiero al introducir procesos de aprendizaje continuo y mejora adaptativa mediante retroalimentación humana (fine tuning).


# 3. Alcance
Considerando el objetivo general y los objetivos específicos planteados, y de acuerdo al flujo de “VeriFactura” (recepción, filtrado, OCR, Modelo entrenado (LLM), resultados), se detallan las siguientes funcionalidades técnicas:
1. **Módulo de Recepción y Pre-procesamiento de Documentos**:
Ingreso de Facturas Multi-Fuente a la API: Facturas vehiculares en diversos formatos: PDF, JPG, PNG, XML.
* **Filtrado y Clasificación de Documentos:** a través de un algoritmo identifica si el documento es un escaneo, fotografía, documento digital (PDF/Word), o texto plano.
* **Derivación al motor correspondiente:** Dirigir el documento al motor de procesamiento apropiado: LLM para texto plano, OCR para imágenes/escaneos.
2. **Módulo de Procesamiento con Lenguaje Natural (LLM) para Texto Plano:**
Extracción de Información: Reconocimiento de entidades nombradas (NER): Identificar y extraer entidades relevantes como:
■	Nombre y RUC del emisor (concesionaria).
■	Nombre y RUC del receptor (banco/cliente).
■	Número de factura.
■	Fecha de emisión.
■	Chasis - VIN (Vehicle Identification Number).
■	Monto total.
■	Impuestos.
■	Marca
■	Tipo
■	Clase
■	Modelo
■	Color
■	Código de motor
■	RAMV
■	Año

* Validación y Normalización de Datos:
    * Validación de formatos: Verificar que los datos extraídos cumplan con formatos esperados (ej. formato del RUC, formato de fecha).
    * Normalización de datos: Estandarizar los datos extraídos (ej. convertir todas las fechas al mismo formato).
3. **Módulo de Procesamiento OCR con Azure Vision Studio:**
**Integración con Azure Vision Studio:**
* Llamada a la API de OCR: Invoca la API de Read de Azure Vision Studio.
* Pre-procesamiento de Imágenes (antes del OCR):
    * Corrección de inclinación: Corregir la inclinación de la imagen para mejorar la precisión del OCR.
    * Eliminación de ruido: Reducir el ruido en la imagen (manchas, sombras) para mejorar la calidad de la imagen para el OCR.
    * Mejora de contraste: Ajustar el contraste de la imagen para facilitar la detección del texto.
* Extracción de Texto:
    * Reconocimiento de texto en tablas: Identifica y extrae texto de tablas en la factura.
    * Reconocimiento de texto en campos clave: Prioriza el reconocimiento de texto en áreas específicas de la factura donde se espera encontrar información crítica.
    * Post-procesamiento del texto OCR:
    * Corrección de errores comunes de OCR: Implementa algoritmos para corregir errores comunes de OCR (sustitución de "O" por "0", "l" por "1").
    * Detección de idioma: Identificar el idioma del texto OCR para facilitar la interpretación semántica.
4. **Módulo de Interpretación Semántica (Post-OCR):**
* Asignación de Campos Clave:
    * Reglas y patrones: Reglas y patrones basados en la disposición típica de las facturas vehiculares para asignar el texto extraído por el OCR a los campos clave (VIN, RUC, Monto total, Fecha).
    * Uso de LLM para mejorar la asignación: Se utiliza LLM para comprender el contexto del texto y asignar los campos clave con mayor precisión, especialmente en casos donde las reglas y patrones no son suficientes.
* Validación y Normalización de Datos (Post-OCR):
    * Validación de formatos: Verificar que los datos asignados a los campos clave cumplan con los formatos esperados.
    * Normalización de datos: Estandarizar los datos extraídos (convertir todas las fechas al mismo formato).
    * Validación cruzada: Comparar los datos extraídos de diferentes partes de la factura para asegurar la consistencia.

**5. Mejora Continua del Modelo:**
- Almacenamiento de datos procesados: Almacenar las facturas procesadas y los datos extraídos para entrenar y mejorar continuamente los modelos de IA.
- Revisión y corrección manual: Permitir la revisión y corrección manual de los datos extraídos por el sistema.
- Retroalimentación al modelo: Utilizar las correcciones manuales para retroalimentar y mejorar los modelos de IA.
**Módulo de Interfaz de Usuario (UI) y Monitoreo:**
* Interfaz de Usuario:
    - Visualización de facturas: Permitir a los usuarios visualizar las facturas procesadas.
    - Edición de datos extraídos: Permitir a los usuarios editar los datos extraídos por el sistema.
## Alcance excluido
Con la intención de mantener el alcance del proyecto enfocado y realista dentro del plazo de 6 semanas no se incluye en este proyecto:

**Funcionalidades Fuera del Alcance de Facturas Vehiculares**
* Procesamiento de Otros Documentos Bancarios: No se incluirá el procesamiento de cheques, extractos bancarios, solicitudes de crédito, u otros documentos bancarios.
* Soporte para Idiomas Adicionales: Inicialmente, “VeriFactura”  no soportará idiomas distintos al español. La expansión a otros idiomas requeriría investigación y entrenamiento adicionales.

**Funcionalidades de Automatización Bancaria Amplia** 
* Aprobación Automática de Créditos: Aunque “VeriFactura” extraerá datos relevantes para la aprobación de créditos, NO tomará decisiones automáticas de aprobación o rechazo. La decisión final permanece en manos de los analistas bancarios.
* Integración Directa con Sistemas Contables: “VeriFactura” no se integrará automáticamente con los sistemas contables de la banca. La exportación de datos estará disponible, pero la importación y reconciliación contable se manejará por separado.

# 4. Cronograma de desarrollo (planificado)
| Sprint | Duración | Sprint Goal | Users Stories | Definition of Done |
| :--------: | :--------: | :--------: |:--------: |:--------: |
| 1 | 1 semana | Arquitectura y entorno listo para pipeline inicia: Detección PDF con/sin texto, OCR vs parse directo. | US-A1 Detectar texto embebido en PDF (3pt) US-A2 Gestor de ingesta y metadatos (5pt) US-B2 Normalizador de texto (5pt) | A1: Precisión ≥95% en set de 100 PDF A2: Cola con reintentos y pruebas de carga mínimas (50 doc/min) B2: Reducción de errores OCR ≥10% vs baseline.|
| 2 | 2 semanas | Pipeline OCR → parser → LLM → validaciones API v0 expuesta | US-B1 Integrar Azure OCR (5pt) US-C1 Extracción LLM de RUC/VIN/totales (8pt) US-C2 Validadores (VIN=17, sumas, RUC) (5pt) US-D1 API REST v0 /extract con auth (5pt) | B1: Inferencia OCR en lote. latencia de ≤3s/página. C1: prompts/few-shot versionados. F1 inicial ≥0.85 en set de validación. C2: Reglas y mensajes de error. Exactitud en sumas ≥0.98 y validador de VIN. D1: Swagger listo. |
| 3 | 2 semanas | Mejorar métricas (tuning) y validar con reportes reproducibles. | US-F1 Script de evaluación (5pt) US-E1 UI simple de carga/resultado (3pt) US-C1b Optimización de prompts por concesionaria (5pt) | F1: reporte reproducible. E1: subir doc, ver JSON y banderas de confianza C1b: ≥0.95 F1 en VIN/totales/RUC. |
| 4 | 1 semana | Pruebas integrales, documentación y demo final lista | US-QA1 Testing integral (funcional/carga) (5pt) US-D3 Hardening y resiliencia (3pt) US-DOC1 Docs técnicas/usuario (3pt) US-DEMO1 Presentación final (2pt) | QA1: Pruebas con 50 facturas, p95 API ≤5s. D3: Reintentos idempotentes, timeouts. DOC1: Guía de despliegue, runbook de incidencias, README endpoints. DEMO1: guión y video corto. |
# 5. Recursos necesarios
## Recursos de hardware
| Recurso | Descripción | Uso Principal | Observaciones |
| :-------- | :-------- | :-------- |:-------- |
| Servidores Cloud (Azure VMs) | Máquinas virtuales de 8–16 vCPUs, 32–64 GB RAM, GPU opcional (NVIDIA T4/A10). | Entrenamiento de modelos (NER, LLM fine-tuning ligero), ejecución OCR masivo, pruebas de estrés. | Escalables bajo demanda. GPU recomendada solo para entrenamiento NER.|
| Storage en la Nube (Blob Storage / Data Lake) | Almacenamiento seguro y estructurado de facturas (PDF, imágenes, XML). | Dataset mixto (real + sintético), logs y outputs estructurados (JSON). | Con encriptación AES-256 y control de acceso RBAC.|
| Entornos de Desarrollo (Dev/QA/Prod) | Ambientes separados en la nube. | Garantizar CI/CD, pruebas y despliegues seguros. | Infra mínima con contenedores (Docker/Kubernetes).|
## Recursos de software
| Recurso | Descripción | Uso Principal | Observaciones |
| :-------- | :-------- | :-------- |:-------- |
| Azure Vision Studio (OCR API) | OCR de alta precisión con pre-procesamiento. | Extracción de texto de PDFs/imágenes. | Pago por consumo (API calls).|
| Azure Cognitive Services (NER + LLM API) | Extracción semántica de entidades. | Identificación de VIN, RUC, montos, fechas, etc. | Posible ajuste con prompts o fine-tuning ligero.|
| Python + Frameworks (FastAPI, Flask) | Backend para pipeline de recepción. | API core de VeriFactura. | Compatible con microservicios.|
| Librerías IA/ML | PyTorch / TensorFlow, Hugging Face, Openai, Pypdf2, Fastapi | Entrenamiento y validación NER, reglas post-OCR. | Uso puntual, no todo desde cero.|

# 6. Riesgos identificados y mitigación
* Riesgo 1: OCR pobre en facturas mal escaneadas.
    * Mitigación: Preprocesos (deskew/denoise).
* Riesgo 2: Variabilidad de plantillas baja recall.
    * Mitigación: Few-shot por concesionaria frecuente + post-proceso por patrones.
* Riesgo 3: Tiempos altos de respuesta.
    * Mitigación: Batching, caché OCR, colas asíncronas y límites de tamaño.
* Riesgo 4: No se alcanza F1 objetivo ≥0.90 en críticos.
    * Mitigación: Híbrido reglas+LLM (regex, checks), más ejemplos y curación de casos difíciles.
* Riesgo 5: Sesgos por concesionaria dominante.
    * Mitigación: Estratificar validación y balancear dataset.
* Riesgo 6: Deuda técnica afecta estabilidad.
    * Mitigación: Bug bash + priorización de hallazgos críticos antes de release.

