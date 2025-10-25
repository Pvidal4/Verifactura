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

Verifactura articula tres familias de modelos para convertir documentos vehiculares en información accionable: 
* Servicio de extracción que administra modelos de lenguaje (LLM).
* Módulo OCR sobre Azure.
* Clasificador Random Forest.
  
**Servicio de extracción que administra modelos de lenguaje (LLM)**

El LLM se invoca con una estrategia de “JSON schema” que obliga a responder únicamente con los campos esperados de una factura vehicular. Para lograrlo, se define un esquema exhaustivo de campos (marca, modelo, VIN, totales, etc.).

Se elaboró un system prompt que orienta al modelo sobre cómo reparar rupturas típicas del OCR y cómo normalizar valores sin espacios. 

La implementación de **OpenAILLMService** encapsula el uso de **Chat Completions de OpenAI**: 
* Asegura credenciales
* Permite ajustar parámetros (temperatura, top-p, razonamiento)
* Cuando hay imágenes, empaqueta texto y píxeles en segmentos compatibles con el modo vision para mejorar la comprensión contextual.

**Módulo OCR sobre Azure**

El **AzureOCRService** es un contenedor ligero sobre **Azure Form Recognizer**:
* Autentica con DocumentAnalysisClient
* Ejecuta el modelo prebuilt-read y concatena todas las líneas detectadas
* Reintentos inteligentes cuando el servidor rechaza el content_type proporcionado. 

El **ExtractionService** reutiliza y cachea instancias OCR, fuerza su uso cuando el archivo es una imagen o cuando el texto crudo está vacío, e incluso aplica estrategias adicionales para PDFs (renderizado de páginas a imágenes) si la lectura directa falla. 

**Clasificador Random Forest**

Verifactura complementa el análisis con un modelo de clasificación Random Forest entrenado a partir de atributos vehiculares (marca, tipo, clase, capacidad, combustible, ruedas y total). 

El pipeline de entrenamiento arma un ColumnTransformer que combina one-hot encoding para categorías y estandarización para numéricos, ajusta un RandomForestClassifier con 400 árboles y guarda el modelo empaquetado junto a métricas para auditoría posterior. 

## 4. 📊 Resultados

**Comparativo de métricas**

En todos los casos el modelo de clasificación utilizado en **Random Forest**

* **Modelo original**

Al estar compuesto por 26 registros, al estratificar los datos, presenta valores ínfimos que no pueden ser reevaluados, detallamos las métricas y la matriz de confusión del mismo:

<img width="639" height="238" alt="image" src="https://github.com/user-attachments/assets/39d5b2c2-7ab3-415b-9573-43d5e4337deb" />

<img width="866" height="611" alt="image" src="https://github.com/user-attachments/assets/9d226942-f2f4-4398-8760-4e9cdd5355c3" />

* **Entrenamiento con data sintética - primera corrida**

Al cambiar a pipeline documental, en esta clasificación la variable a predecir es la "Categoría del cliente"

<img width="713" height="248" alt="image" src="https://github.com/user-attachments/assets/4afba7ff-15c0-4f9a-a40c-6ec15d8910da" />

<img width="886" height="647" alt="image" src="https://github.com/user-attachments/assets/9725013b-edbd-4cbe-81ea-38b7293fbe9b" />

* **Modelo final**

Una vez entrenado y conservando la estructura de pipeline documental, se detallan las métricas finales:

<img width="886" height="543" alt="image" src="https://github.com/user-attachments/assets/6df35584-2292-4918-a3c1-2245c6f83cf2" />





  

**Detalle de rendimiento de los tiempos**

Según el modelo LLM seleccionado, estos son los tiempos de respuesta promedio:
- gpt-5 (reasoning HIGH) → 54s
- gpt-5 (reasoning MEDIUM) → 26s
- gpt-5 (reasoning LOW) → 14s
- gpt-5-mini (reasoning LOW) → 8s
- gpt-4.1-mini → 5s

## 5.🔑Instalación y uso

# Guía de instalación de Verifactura

## Requisitos del sistema
- **Sistema operativo:** Windows 10/11, macOS 13+ o cualquier distribución Linux actual.
- **Python:** 3.12 con `pip` actualizado.
- **CPU:** Procesador de 4 núcleos (se recomienda soporte para instrucciones AVX).
- **Memoria RAM:** 8 GB mínimo (16 GB recomendados para OCR y modelos locales).
- **GPU (opcional):** NVIDIA con soporte CUDA 12.1 para acelerar modelos locales con PyTorch.
- **Credenciales externas:**
  - Clave de API de OpenAI para el flujo de extracción vía API.
  - Endpoint y clave de Azure Form Recognizer para OCR.

## Instrucciones paso a paso para instalar
1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/<tu-organizacion>/Verifactura.git
   cd Verifactura
   ```
2. **Crear y activar un entorno virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows usa: .venv\\Scripts\\activate
   ```
3. **Instalar dependencias principales**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Las bibliotecas críticas como `fastapi` 0.110.0, `openai` 1.17.0, `PyPDF2` 3.0.1, `azure-ai-formrecognizer` 3.3.0 y `transformers` 4.39.3 están listadas en `requirements.txt`.
4. **(Opcional) Acelerar modelos locales con GPU**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Este comando instala las variantes de PyTorch compatibles con CUDA 12.1. Si no cuentas con GPU NVIDIA, conserva la instalación CPU incluida en `requirements.txt`.


5. **Instalación de Microsoft Visual C++ Redistributable.**

Puedes descargarlo en https://aka.ms/vs/17/release/vc_redist.x64.exe  
   
6. **Configurar variables de entorno**
   - Copia `.env.example` a `.env` (si existe) o crea un archivo `.env` con las claves siguientes:
     ```env
     OPENAI_API_KEY=tu_clave
     OPENAI_MODEL=gpt-5-mini
     AZURE_FORM_RECOGNIZER_ENDPOINT=https://<tu-endpoint>.cognitiveservices.azure.com/
     AZURE_FORM_RECOGNIZER_KEY=tu_clave_azure
     ```
   - Ajusta `LOCAL_LLM_MODEL_PATH` o `LOCAL_LLM_MODEL_ID` si cuentas con pesos locales.

## Comandos para ejecutar el proyecto
- **API en modo desarrollo**
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```
- **Pruebas automatizadas**
  ```bash
  pytest
  ```
  Consulta la guía detallada en [`docs/Pruebas.md`](Pruebas.md) para conocer
  opciones adicionales y recomendaciones de ejecución.
- **Reentrenar el modelo Random Forest**
  ```bash
  python -m train.random_forest
  ```
  (Asegúrate de que `train/data/verifactura_dataset.csv` contenga tu dataset actualizado.)

## Ejemplos de uso
- **Extracción desde texto plano**
  ```bash
  curl -X POST http://localhost:8000/api/v1/extract/text \
       -H "Content-Type: application/json" \
       -d '{
             "text": "Factura Nº 123 emitida el 15/03/2024...",
             "llm_provider": "api",
             "llm_model": "gpt-5-mini"
           }'
  ```
- **Extracción desde archivo PDF**
  ```bash
  curl -X POST http://localhost:8000/api/v1/extract/file \
       -F "file=@factura.pdf" \
       -F "force_ocr=false"
  ```
- **Clasificación de categoría vehicular**
  ```bash
  curl -X POST http://localhost:8000/api/v1/predictions \
       -H "Content-Type: application/json" \
       -d '{
             "marca": "TOYOTA",
             "tipo": "SUV",
             "clase": "CAMIONETA",
             "capacidad": 5,
             "combustible": "GASOLINA",
             "ruedas": 4,
             "total": 24990.00
           }'
  ```

## 6. 💻 Interfaz de usuario

## Descripción de la interfaz
La interfaz HTML (`app/templates/index.html`) ofrece una experiencia en una sola página orientada a asistentes de verificación vehicular:
- **Encabezado contextual:** muestra la identidad de Verifactura, accesos rápidos y un indicador de modo desarrollador.
- **Selector de modo de extracción:** permite alternar entre carga de texto, archivos únicos o lotes de documentos.
- **Zona de arrastre y carga:** acepta PDF, XML, JSON o imágenes para activar OCR y visualizar metadatos clave.
- **Editor de texto enriquecido:** facilita pegar contenido manualmente y habilitar opciones avanzadas del LLM (modelo, temperatura, top-p, esfuerzo de razonamiento).
- **Panel de resultados:** divide la salida en vista estructurada (campos extraídos), vista JSON y vista previa del archivo original.
- **Módulo de predicción:** ejecuta el clasificador Random Forest para estimar la categoría vehicular y mostrar probabilidades.

Todos los componentes siguen un estilo consistente basado en variables CSS declaradas en la cabecera del documento.

## Instrucciones para usar la interfaz
1. **Acceder a la aplicación**: abre `http://localhost:8000` en un navegador moderno (Chrome, Edge o Firefox).
2. **Seleccionar el tipo de entrada**:
   - Pestaña *Texto* para pegar contenido directamente.
   - Pestaña *Archivo* para subir un solo PDF/XML/JSON.
   - Pestaña *Lote* para procesar múltiples archivos de forma secuencial.
3. **Configurar parámetros opcionales**:
   - Activa el *Modo desarrollador* para visualizar campos avanzados y credenciales temporales.
   - Ajusta proveedor/modelo LLM, temperatura y top-p según el grado de creatividad deseado.
   - Define proveedor OCR (Azure) y, si es necesario, sobrescribe endpoint/clave.
4. **Lanzar la extracción**: pulsa **Procesar**; el panel de resultados mostrará:
   - Campos clave resaltados y normalizados.
   - JSON descargable con los datos estructurados.
   - Vista previa del archivo original en un modal.
5. **Obtener predicción de categoría**: completa el formulario del módulo *Clasificador vehicular* y presiona **Calcular categoría** para visualizar la etiqueta estimada y la distribución de probabilidades.
6. **Descargar o reiniciar**: usa los botones de exportación para guardar el JSON o limpia la sesión con **Restablecer**.

## 7. 🔩 Estructura del proyecto
El repositorio de Verifactura está constituido por las siguientes carpetas:

**app:** desarrollo de las aplicaciones utilizadas en Verifactura.

**docs:** dentro de esta carpeta se encuentra documentado el paso a paso del proyecto
* **planificacion:** planteamiento del problema, objetivos, cronograma, recursos y riesgos identificados en la etapa inicial.
* **análisis_datos:** detalle de la composición del Dataset_inicial, análisis exloratorio, estadística descriptiva, análisis bivariado, outliers, matriz de correlaciones.  
* **arquitectura:** routers, extraction service, OpenIA LLM Service, Prediction Service 
* **optimización:** detalle de data de entrenamiento, definición de hiperparámetros, análisis de sensibilidad, partial dependece plots, ranking de hiperparámetros, análisis de interacciones. 
* **consideraciones éticas:** análisis de sesgo, riesgos identificados y medidas de mitigación, impacto social positivo y negativo, uso y mal uso de Verifactura, limitaciones.
* **manual de usuario:** guía paso a paso para usar la interfaz, capturas de pantallas anotadas, explicación de cada funcionalidad, troubleshooting (problemas comunes y soluciones), preguntas frecuentes (FAQ), información de contacto para soporte.

**imagenes:** carpeta con todos los gráficos y tablas obtenidas en los diferentes procesos de construcción de Verifactuta.

**models:** carpeta con los tres modelos utilizados LLM, OCR, Random Forest. 

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
  
### Uso dual y mal uso de Verifactura

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

Copyright 2025 - UEES.

## 11. 🤝 Agradecimientos y referencias

Toda nuestra gratitud a los compañeros y maestros que nos acompañaron en el proceso, especialmente al Grupo 1.
