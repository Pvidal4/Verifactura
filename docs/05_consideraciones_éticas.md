# 1. Análisis de sesgo
## Sesgo de género y ocupacional
Al tratarse de datos transaccionales no existen sesgos demográficos, de género o culturales dentro del Dataset, sin embargo, en la evaluación de riesgos, se detectó *sesgo de género  ocupacional*, pues la implementación del sistema puede reproducir o amplificar desigualdades existentes al automatizar tareas predominantemente ocupadas por mujeres (digitación y verificación). Dado que los roles asociados a la tecnología (STEM) son mayoritariamente masculinos, el despliegue de **VeriFactura** podría acentuar la brecha de género en el empleo financiero.
El impacto negativo identificado en el proyecto muestra que las mujeres representan solo el 25% de trabajadoras en áreas STEM en Ecuador, mientras que ocupan en mayor proporción los cargos administrativos que serán sustituidos por la automatización.

**Grupo afectado:** Personal administrativo - operativo (principalmente mujeres).

# 2. Equidad y fairness
## Análisis de Equidad
**VeriFactura** tiende a reducir desigualdades operativas entre las concesionarias y la banca, al permitir procesos más rápidos, precisos y trazables. Sin embargo, a nivel laboral y de género puede aumentar desigualdades preexistentes, especialmente para las personas (en su mayoría mujeres) que ocupan puestos de digitación y verificación, los cuales son los más afectados por la automatización. Podemos concluir que:

* Reduce desigualdades entre actores institucionales (banca–proveedores).
* Aumenta desigualdades dentro de la fuerza laboral, particularmente en grupos con menor acceso a formación tecnológica o reconversión laboral.
## Distribución de beneficios
Los principales beneficiarios son las instituciones financieras y concesionarias; los beneficios para el personal operativo son limitados o dependen de políticas de transición laboral efectivas.

| Grupo | Beneficio Directo/Indirecto | Nivel de beneficio | Observación |
| :--------: | :--------: | :--------: | :--------: |
| Alta gerencia / Institución financiera | Reducción de costos operativos (−95% de carga manual). | Alto | Obtienen beneficios económicos tangibles y reputacionales. |
| Concesionarias | Procesamiento rápido y menos rechazos (−90% en tiempos). | Alto | Beneficio en eficiencia y satisfacción del cliente. |
| Clientes finales (compradores de vehículos) | Aprobaciones más rápidas y confiables. | Medio | Beneficio indirecto en tiempos, pero sin impacto directo en equidad. |
| Digitadores y verificadores | Liberación de tareas repetitivas; potencial reconversión. | Bajo / Condicionado | Beneficio depende de acceso a capacitación o reposicionamiento. |
| Sociedad y medio ambiente | Menor consumo de papel y huella ambiental. | Medio | Contribución positiva a sostenibilidad, aunque no equitativa individualmente. |

El acceso al sistema no es equitativo:

* **Instituciones y concesionarias** disponen de infraestructura digital y capacidades técnicas para integrarse fácilmente.
* **Empleados operativos** carecen de igual acceso a las herramientas tecnológicas o programas de reentrenamiento, lo que limita su participación en la nueva etapa del proceso.
* **Clientes finales** acceden solo a los resultados del sistema, no a su funcionamiento o datos subyacentes.
## Estrategia de mitigación
**Estrategia:** Implementar monitoreo de sesgo de género en la automatización y establecer un programa de reconversión laboral y capacitación digital para los grupos más afectados (particularmente mujeres en roles de digitación).
Tipo: Política, Educación
**Implementación:** 
* Aplicar auditorías de equidad laboral y de impacto de automatización con enfoque de género.
* Diseñar e impartir programas de capacitación en herramientas digitales, IA y control de calidad documental.
* Establecer cuotas de participación equitativa en nuevas funciones relacionadas con el sistema (supervisión, validación, soporte).

**Cuándo:** Antes del despliegue y de forma continua durante la adopción del sistema.

**Quién es responsable:** Gerencia de Talento Humano, Innovación / Ética Digital.

**Efectividad esperada:** Alta



# 3. Privacidad
## Riesgo identificado: Uso y resguardo de datos sensibles de facturas
VeriFactura procesa facturas digitales que contienen datos personales y financieros de personas naturales (compradores de vehículos), pero los excluye del análisis, no así a la información comercial de las concesionarias. Si no se establecen controles estrictos, podría existir riesgo de re-identificación o acceso no autorizado a datos sensibles.

**Evidencia:** El sistema realiza extracción automática de entidades como RUC, VIN, subtotal, IVA, total, entre otros, de los cuales se podría inferir algún dato personal como identificación y nombre, lo que debe ser restringido según la Ley Orgánica de Protección de Datos Personales (LOPDP, Ecuador).

**Severidad:** Alta.

**Grupo afectado:** Clientes finales, concesionarias y entidades financieras.

## Estrategia de mitigación
**Estrategia:** Adoptar un marco de gobernanza de datos personales que incluya encriptación, anonimización y controles de acceso basados en roles, conforme a la Ley Orgánica de Protección de Datos Personales (LOPDP).

**Tipo:** Técnica, Política

**Implementación:**
* Aplicar encriptación AES-256 y anonimización de datos personales (cedula/RUC, nombre) antes del procesamiento.
* Definir políticas de retención y eliminación segura de datos, y registrar auditorías de acceso.
* Obtener consentimiento informado y notificar a los titulares del tratamiento automatizado.
**Cuándo:** Previo al lanzamiento y de manera continua durante la operación del sistema.
**Quién es responsable:** Área de Tecnología (TI), DPO (Oficial de Protección de Datos), Cumplimiento Normativo.

**Efectividad esperada:** Alta

# 4. Transparencia y explicabilidad
## Riesgo: Falta de explicabilidad y transparencia del modelo

El proceso de extracción y validación automatizada se basa en modelos de IA que podrían no ser fácilmente interpretables por usuarios no técnicos. Esto puede limitar la comprensión de por qué una factura es aceptada o rechazada, reduciendo la confianza de los operadores humanos y de las concesionarias.

**Evidencia:** En la etapa actual, los algoritmos de reconocimiento y validación no incluyen mecanismos de explainability visibles (p. ej., reportes interpretables o trazabilidad de decisión). Tampoco se especifican canales para que los usuarios comprendan las razones de error o rechazo.

**Severidad:** Media.

**Grupo afectado:** Concesionarias, digitadores/verificadores, área de operaciones.

## Estrategia de mitigación
**Estrategia:** Incorporar herramientas de explicabilidad y trazabilidad de decisiones (por ejemplo, LIME o SHAP) y desarrollar una interfaz de usuario con reportes interpretables para las concesionarias y personal operativo.

**Tipo:** Técnica, Diseño
**Implementación:**
* Integrar módulos de interpretación que muestren las entidades clave que influyeron en la validación o rechazo de una factura.
* Desarrollar paneles de trazabilidad con logs de decisiones accesibles para auditores y usuarios autorizados.
* Crear documentación de funcionamiento y limitaciones del modelo, disponible en lenguaje claro.

**Cuándo:** Antes del lanzamiento (fase piloto) y actualizaciones trimestrales.

**Quién es responsable:** Científico Datos y UX, con revisión de TI e Innovación.

**Efectividad esperada:** Media-Alta


# 5. Impacto social
## Impactos positivos
1.	**Eficiencia mejorada**
Verifactura asegura que el tiempo de procesamiento por factura se mantenga entre 5 y 30 segundos, garantizando una respuesta en tiempo real para soportar la operación bancaria continua. 
De este proceso se benefician: concesionarias, clientes finales (compradores de vehículos que aspiran a un crédito), al obtener una respuesta con un tiempo 90% menor  al que se obtiene actualmente.
Así mismo, la banca local reduce sus costos operativos al reducir la operativa manual en un 95%.   
2.	**Mejora la calidad de vida laboral**
Los empleados inmersos en el ciclo de vida operativo (digitadores, verificadores), quedarán liberados de actividades tediosas y repetitivas, y podrán enfocarse a tareas actividades más estratégicas y creativas que requieren habilidades humanas como la revisión de casos específicos que son derivadas por el sistema al no cumplir el estándar de precisión definido (95%). 
3.	**Nuevas oportunidades de ahorro en recursos físicos**
Puesto que Verifactura recibe exclusivamente documentación digital se reduce el uso de documentos físicos, generando un ahorro de papel y reprocesos,  apalancando los estándares de sostenibilidad que deben cumplir normativamente los bancos locales.

## Impactos negativos
1.	**Desplazamiento laboral**
Al automatizarse los procesos de extracción (ya no será necesario digitar), y verificación (lo realizará la app), los digitadores y verificadores verán reducido su número, y sólo intervendrán en los casos en los que no se cumpla el estándar de precisión (human-in-the-loop). 
Este riesgo tiene impacto y probabilidad Altos, pues es consecuencia directa de la automatización del proceso a mejorar.  
2.	**Erosión de autonomía**
Se crea una dependencia tecnológica, y se generan nuevos riesgos de ciberataques, por ende debe considerarse la seguridad como parte de las verticales a implementar.  
Este riesgo tiene impacto Alto y probabilidad Media, pues existen medidas de mitigación como software especializado, así como protocolos de accesibilidad, disponibilidad y confidencialidad.
3.	**Aumento en la brecha de habilidades**
Existe un mayor riesgo de pérdida de empleo para las mujeres, debido a que esta automatización es más agresiva en los roles que ellas ocupan en mayor proporción (digitación y verificación), en contraste los hombres están más asociados con habilidades que generan automatización, como las STEM (Ciencia, Tecnología, Ingeniería y Matemáticas).
Este riesgo tiene impacto y probabilidad Altos, pues en Ecuador las mujeres representan el 25% de trabajadoras de STEM.  

# 6. Responsabilidad
## Cadena de Responsabilidad
| Rol | Responsabilidades | Rendición de cuentas |
| :--------: | :--------: | :--------: |
| TI / Innovación (Pedro Vidal, Andrea Morán) | Implementación técnica correcta, integración segura, encriptación y anonimización de datos según la Ley Orgánica de Protección de Datos Personales (LOPDP, 2021). | Code reviews, pruebas de ciberseguridad, auditorías de acceso y logs técnicos. |
| Oficial de Protección de Datos (DPO) | Supervisar el cumplimiento de la LOPDP: consentimiento informado, limitación de finalidad y minimización de datos.| Auditorías de privacidad semestrales, reportes de cumplimiento ante la Superintendencia de Protección de Datos. |
| Gerencia de Talento Humano | Diseñar e implementar programas de reconversión laboral y capacitación digital, en concordancia con los artículos 1, 5 y 42 del Código de Trabajo.| Evidencia de programas de capacitación, reportes de participación y métricas de reinserción laboral. |
| Alta Gerencia / Organización | Establecer políticas de gobernanza, asignar recursos y asegurar supervisión ética de IA.| Políticas internas aprobadas, seguimiento por comité de ética digital, evaluación anual de impacto social. |
# 7. Uso dual y mal uso
## Uso dual
Como todo sistema basado en inteligencia artificial que automatiza el procesamiento de información sensible, *VeriFactura* posee un potencial de uso dual, es decir, podría ser empleado con fines distintos a los originalmente previstos, tanto positivos como negativos.
Aunque fue diseñado para optimizar la gestión documental (extracción y validación) de facturas vehiculares, su arquitectura podría adaptarse para otros contextos en los que la extracción masiva de datos financieros o comerciales derive en vulneraciones éticas o legales.
Escenarios:

- **Reutilización del modelo** de extracción para procesar documentos personales o contractuales (nóminas, comprobantes de pago, escrituras, etc.) sin consentimiento explícito de los titulares.
- **Integración con sistemas de vigilancia o scoring crediticio** sin supervisión ética, lo cual podría derivar en prácticas discriminatorias o violaciones a la privacidad.
- **Transferencia o entrenamiento** secundario del modelo con datos no anonimizados, generando un riesgo de reidentificación o de sesgo no controlado.

Para prevenir estos escenarios, la gobernanza de uso debe estar explícitamente limitada al dominio financiero vehicular, y cualquier expansión del alcance debe requerir:
- Aprobación del Comité de Ética Digital y del Oficial de Protección de Datos (DPO).
- Revisión técnica y legal previa a todo cambio de propósito de uso.
- Registro documentado de nuevos datasets y objetivos de tratamiento, conforme a la LOPDP (2021).

## Mal uso o uso indebido
El riesgo de mal uso se presenta cuando la herramienta se implementa, manipula o configura fuera de los controles previstos. Entre los posibles escenarios se identifican:
- Uso por personal no autorizado que acceda a datos sensibles o modifique parámetros del modelo sin registro en los logs.
- Elusión de protocolos de revisión humana, confiando plenamente en los resultados automáticos sin control de precisión o trazabilidad.
- Manipulación intencionada de facturas digitalizadas para obtener beneficios indebidos (por ejemplo, validaciones falsas o fraude documental).
**Medidas preventivas:**
- Control de acceso basado en roles (RBAC) y autenticación multifactor.
- Auditorías trimestrales de trazabilidad de decisiones y revisiones de sesgo.
- Formación ética continua del personal sobre los límites y responsabilidades en el uso de IA.
- Implementación de alertas automáticas ante intentos de acceso o modificación no autorizada.

# 8. Limitaciones reconocidas
## Limitaciones sociales y éticas
- **Reconversión laboral incompleta:** El éxito de las medidas de mitigación dependerá de la ejecución real de los programas de capacitación, los cuales no garantizan por sí mismos la recolocación laboral.
- **Brecha de acceso digital:** No todos los grupos afectados (especialmente mujeres y personal administrativo tradicional) cuentan con igual acceso a recursos tecnológicos o programas de actualización.

- **Dependencia institucional:** La operación del sistema depende de la infraestructura tecnológica y del soporte continuo del proveedor, lo que podría generar vulnerabilidad ante fallas o desactualizaciones.
