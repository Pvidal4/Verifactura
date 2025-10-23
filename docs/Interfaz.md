# Interfaz web de Verifactura

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
