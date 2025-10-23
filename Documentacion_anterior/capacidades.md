# Capacidades y Funcionalidades de Claude Code - Documentación Completa

## 1. Descripción General

### 1.1 ¿Qué es Claude Code?
Claude Code es una herramienta de codificación agentiva basada en IA que vive en tu terminal, comprende tu base de código y te ayuda a programar más rápido ejecutando tareas rutinarias, explicando código complejo y manejando flujos de trabajo de git, todo a través de comandos en lenguaje natural.

### 1.2 Características Principales
- **Terminal nativo**: Funciona directamente en la línea de comandos (bash, zsh, fish, PowerShell)
- **Comprensión de contexto**: Entiende la estructura y dependencias del proyecto
- **Acciones directas**: Puede leer, editar, crear archivos y ejecutar comandos
- **Integración Git**: Maneja flujos de trabajo completos de control de versiones
- **Búsqueda instantánea**: Busca en bases de código de millones de líneas al instante

## 2. Instalación y Configuración

### 2.1 Requisitos del Sistema
- **Node.js**: Versión 18 o superior
- **Sistemas operativos**: Compatible con Windows, macOS y Linux
- **Terminal**: Cualquier terminal compatible con Node.js

### 2.2 Métodos de Instalación

#### Instalación via NPM (Principal)
```bash
npm install -g @anthropic-ai/claude-code
```

#### Otros métodos disponibles
- pip (Python)
- Homebrew (macOS)
- Extensión VS Code (desde el marketplace)

### 2.3 Configuración Inicial
1. Navegar al directorio del proyecto: `cd tu-proyecto`
2. Iniciar Claude Code: `claude`
3. Autenticación con cuenta Claude existente o API key

## 3. Modos de Operación

### 3.1 Modo por Defecto
- Claude sugiere cambios y espera aprobación antes de ejecutar
- Requiere confirmación explícita para modificaciones de archivos
- Ideal para control total sobre las operaciones

### 3.2 Modo Auto
- Claude trabaja en el código y edita archivos sin esperar permisos
- Ejecución automática de tareas aprobadas
- Útil para tareas repetitivas y bien definidas

### 3.3 Modo Plan
- Activación: Presionar Shift+Tab dos veces
- Separa investigación y análisis de la ejecución
- Claude no editará archivos ni ejecutará comandos hasta aprobar el plan
- Pensamiento extendido para crear estrategias comprehensivas
- Mejora significativa en seguridad

### 3.4 Modo Dangerously Skip Permissions
```bash
claude --dangerously-skip-permissions
```
- Omite todas las verificaciones de permisos
- Claude trabaja sin interrupciones hasta completar
- Ideal para corregir errores de linting o generar código boilerplate

### 3.5 Modo Thinking (Pensamiento)
- Activación: Usar palabras clave como "think", "think harder", o "ultrathink"
- Claude entra en modo de pensamiento profundo antes de actuar
- Genera planes más detallados y considerados
- Útil para problemas complejos que requieren análisis profundo

### 3.6 Modo Headless (Sin interfaz)
```bash
claude -p "tu prompt aquí" --output-format stream-json
```
- Operación no interactiva para CI/CD, scripts de build, y automatización
- Flag `-p` con un prompt habilita el modo headless
- `--output-format stream-json` para salida JSON en streaming
- Ideal para pre-commit hooks y procesos automatizados

## 4. Comandos Disponibles

### 4.1 Comandos Básicos

| Comando | Función |
|---------|---------|
| `/clear` | Limpia el historial de conversación y reinicia el contexto |
| `/resume` | Restaura el estado de conversación anterior |
| `/compact` | Resume la conversación preservando contexto específico |
| `/bug` | Reporta problemas directamente desde Claude Code |
| `/help` | Muestra ayuda y comandos disponibles |
| `/doctor` | Comando de diagnóstico para verificar el estado del sistema |
| `/add-dir` | Agrega directorios con soporte de autocompletado |
| `/plugin` | Instala y gestiona plugins de Claude Code |

### 4.2 Comandos de Configuración

| Comando | Función |
|---------|---------|
| `/output-style` | Configura las preferencias de explicación para las sugerencias |
| `/permissions` | Define qué comandos bash se ejecutan sin solicitar aprobación |
| `/settings` | Accede a configuraciones generales |
| `/mcp` | Verifica el estado del servidor Model Context Protocol |

### 4.3 Comandos de Proyecto

| Comando | Función |
|---------|---------|
| `/init` | Genera archivo Claude.md desde el análisis del proyecto |
| `/agents` | Crea sub-agentes especializados para tareas específicas |
| `/hooks` | Establece triggers de automatización en puntos del ciclo de vida |
| `/review` | Comando personalizado para auditorías de calidad de código |

### 4.4 Comandos de Generación de Código

| Comando | Función |
|---------|---------|
| `claude-code generate` | Crea nuevas funciones, componentes o endpoints |
| `claude-code refactor` | Moderniza y optimiza código existente |
| `claude-code review` | Analiza código con enfoque en seguridad y calidad |
| `claude-code create` | Construye componentes para frameworks |
| `claude-code modify` | Agrega características a archivos existentes |
| `claude-code batch` | Procesa múltiples archivos con coincidencia de patrones |

## 5. Capacidades de Operación con Archivos

### 5.1 Lectura de Archivos
- Lee archivos de cualquier tamaño con paginación inteligente
- Soporte para múltiples formatos (código, markdown, JSON, etc.)
- Comprensión de estructura y dependencias del proyecto

### 5.2 Edición de Archivos
- Edición directa de archivos existentes
- Preservación de formato e indentación
- Ediciones multi-archivo con comprensión de dependencias
- Actualizaciones recursivas automáticas de archivos dependientes

### 5.3 Creación de Archivos
- Generación de nuevos archivos y directorios
- Creación de estructuras de proyecto completas
- Plantillas personalizadas por tipo de archivo

### 5.4 Gestión de Archivos
- Operaciones de movimiento y renombrado
- Eliminación segura con confirmación
- Búsqueda y reemplazo global

## 6. Integración con Git

### 6.1 Operaciones Básicas
- `git status`, `git diff`, `git log`
- Creación y gestión de commits
- Creación y cambio de ramas
- Gestión de stash

### 6.2 Flujos de Trabajo Avanzados
- Creación de pull requests
- Resolución de conflictos de merge
- Búsqueda en historial de commits
- Git worktrees para desarrollo paralelo
- Branch-aware memory banking

### 6.3 Automatización Git
- Generación automática de mensajes de commit
- Pre-commit y pre-push hooks
- Revisión automatizada de pull requests
- Integración con GitHub/GitLab CI/CD

## 7. Sub-Agentes y Especialización

### 7.1 Concepto de Sub-Agentes
- Asistentes AI especializados con sus propias instrucciones
- Ventanas de contexto aisladas
- Permisos de herramientas específicos
- Cada sub-agente opera en su propio contexto, evitando contaminación de la conversación principal
- Se mantienen enfocados en objetivos de alto nivel

### 7.2 Tipos de Sub-Agentes Disponibles
- **oracle**: Agente de consulta y conocimiento
- **code-reviewer**: Revisión especializada de código
- **SDK experts**: Expertos en SDKs específicos
- **domain specialists**: Especialistas en dominios particulares
- **test-runner**: Ejecutor de pruebas automatizadas
- **documentation-writer**: Generador de documentación

### 7.3 Invocación de Sub-Agentes
- **@-mentions**: Usa `@<nombre-del-agente>` para invocar agentes específicos
- Soporte de autocompletado con typeahead para agentes personalizados
- Los sub-agentes heredan todas las herramientas MCP disponibles cuando no se especifica el campo tools

### 7.4 Consolidación de Sub-Agentes
- Combinación eficiente de múltiples agentes
- Orquestación de tareas complejas
- Gestión de contexto entre agentes
- Usa `/agents` para modificar acceso a herramientas con interfaz interactiva

## 8. Agent Skills (Habilidades de Agente)

### 8.1 Concepto de Agent Skills
- Configuraciones controladas por el modelo que habilitan tareas especializadas
- Introducidas en la versión 2.0.20 de Claude Code
- Hacen que la codificación agentiva sea realmente divertida y eficiente
- Compartidas activamente por la comunidad

### 8.2 Características de Agent Skills
- Habilidades específicas del dominio
- Configuraciones preestablecidas para tareas comunes
- Activación mediante comandos o contexto
- Integración con sub-agentes para mayor especialización

### 8.3 Desarrollo de Skills Personalizadas
- Creación de skills propias para casos de uso específicos
- Compartir skills con la comunidad
- Integración con plugins para distribución

## 9. Sistema de Documentación (CLAUDE.md)

### 9.1 Archivos CLAUDE.md
- Documentación de proyecto que persiste entre sesiones
- Almacena convenciones, decisiones y contexto
- Estructura jerárquica en múltiples niveles de directorio
- Claude lee estos archivos al inicio de cada sesión

### 9.2 Niveles de Configuración
1. **Global**: `~/.claude/CLAUDE.md` - Preferencias globales
2. **Organización**: Estándares de la organización
3. **Proyecto**: Guías específicas del proyecto
4. **Directorio**: Configuraciones por carpeta

### 9.3 Sintaxis y Características
- Referencias externas con `@path/to/file`
- Instrucciones específicas del dominio
- Plantillas y snippets reutilizables
- Los archivos se pueden incluir en control de versiones para compartir con el equipo

## 10. Hooks y Automatización

### 10.1 Tipos de Hooks

| Hook | Trigger |
|------|---------|
| PreToolUse | Antes de ejecutar herramientas |
| PostToolUse | Después de ejecutar herramientas |
| Notification | Cuando Claude envía alertas |
| Stop | Al completar tareas |
| Sub-agent Stop | Al finalizar sub-agentes |
| UserPromptSubmit | Al enviar prompts del usuario |
| SessionStart | Al iniciar una nueva sesión |

### 10.2 Casos de Uso de Hooks
- Ejecución automática de tests
- Linting y formateo de código
- Actualización de statusline
- Notificaciones personalizadas
- Validación de seguridad
- Garantizan automatización que no depende de que Claude "recuerde" hacer algo

### 10.3 Configuración de Hooks
- Scripts shell personalizados definidos por el usuario
- Integración con herramientas externas
- Cadenas de hooks para flujos complejos
- Se ejecutan automáticamente en puntos específicos del ciclo de vida

## 11. Model Context Protocol (MCP)

### 11.1 Funcionalidad MCP
- Claude Code funciona como servidor y cliente MCP
- Conexión a múltiples servidores MCP simultáneamente
- Acceso a herramientas externas

### 11.2 Integraciones MCP Disponibles
- **brave-search**: Consultas web
- **Jira**: Gestión de tickets
- **GitHub**: Integración con repositorios
- **Bases de datos**: Acceso directo a datos
- **AWS Services**: Integración con servicios cloud

### 11.3 Configuración MCP
- En configuración de proyecto (disponible al ejecutar en ese directorio)
- Configuración global para todos los proyectos
- Configuración por entorno
- Archivo `.mcp.json` versionable para compartir con el equipo
- Soporte para transportes stdio, SSE, y HTTP streaming
- Soporte para servidores MCP remotos

## 12. Testing y Calidad de Código

### 12.1 Capacidades de Testing
- Configuración de infraestructura completa de testing
- Unit tests, component tests, integration tests
- Escenarios E2E (End-to-End)
- Generación automática de casos de prueba

### 12.2 Ejecución y Corrección
- Ejecuta tests y corrige casos fallidos
- Aplica reglas de linting
- Optimización de cobertura de código
- Target de cobertura específicos

### 12.3 Análisis de Calidad
- Revisión de 6 aspectos para análisis profundo
- Criterios UI/UX en revisiones de diseño
- Escaneo de vulnerabilidades
- Detección de secretos con remediación

## 13. Gestión de Contexto

### 13.1 Ventana de Contexto
- **Capacidad actual**: 200,000 tokens (Claude Code)
- **Próximamente**: 1 millón de tokens
- Auto-compresión cuando se acerca al límite
- Estimación: Las obras completas de Shakespeare caben en el contexto

### 13.2 Estrategias de Gestión
- Usar `/clear` frecuentemente entre tareas
- Scopear chats a características únicas
- Limpiar contexto al completar features
- Context priming para configuración comprehensiva

### 13.3 Branch-Aware Memory
- Memoria específica por rama git
- Persistencia de contexto entre cambios de rama
- Lazy-loading de contexto relevante

## 14. Optimización y Rendimiento

### 14.1 Características de Rendimiento
- Caché de respuestas con TTL configurable
- Procesamiento batch paralelo (4 operaciones concurrentes por defecto)
- Procesamiento optimizado para proyectos grandes
- Compresión de red y persistencia de conexión
- Gestión de colas y límites de operaciones concurrentes

### 14.2 Optimización de Código
- Reducción de tamaño de bundle
- Optimización de performance de queries
- Mejora de eficiencia de renderizado
- Métricas antes/después medibles

### 14.3 Configuración de Rendimiento
```json
{
  "maxConcurrent": 4,
  "cacheEnabled": true,
  "cacheTTL": 3600,
  "compressionEnabled": true
}
```

## 15. Integraciones IDE y Herramientas

### 15.1 VS Code Extension (Beta)
- Experiencia IDE nativa sin requerir terminal
- Instalación desde el marketplace
- Codificación con Claude directamente en el sidebar
- Keybindings personalizables

### 15.2 Vim/Neovim Plugin
- Integración completa con el editor
- Comandos vim personalizados
- Navegación eficiente de sugerencias

### 15.3 JetBrains IDEs
- Soporte para IntelliJ IDEA, PyCharm, WebStorm
- Integración con herramientas de refactoring
- Sincronización con inspecciones del IDE

## 16. CI/CD y DevOps

### 16.1 GitHub Actions
- Generación de workflows completos
- Validación de pull requests
- Despliegue a staging y producción
- Estrategias zero-downtime

### 16.2 GitLab CI/CD
- Configuración de pipelines
- Integración con GitLab runners
- Despliegue automatizado

### 16.3 Integración con Plataformas Cloud
- **Amazon Bedrock**: Despliegue seguro y compatible
- **Google Vertex AI**: Integración empresarial
- **AWS Services**: Conexión con servicios AWS

## 17. Seguridad y Privacidad

### 17.1 Medidas de Seguridad
- Almacenamiento de credenciales en keychains del sistema
- Rotación regular de claves
- Sanitización de prompts antes de transmisión
- Audit trails para cambios generados por IA
- Procesamiento local cuando está disponible

### 17.2 Protección de Datos
- Períodos de retención limitados para datos sensibles
- Acceso restringido a información de sesión
- Políticas explícitas contra uso en entrenamiento de modelos
- Exclusión de archivos sensibles vía `.claude-code-ignore`

### 17.3 Validación de Seguridad
- Escaneo de vulnerabilidades
- Detección de secretos
- Auditoría de dependencias
- Validación de políticas de seguridad

## 18. Flujos de Trabajo Avanzados

### 18.1 Metodologías Estructuradas

#### AB Method
- Workflow basado en especificaciones
- Transforma problemas grandes en misiones enfocadas
- Usa sub-agentes especializados

#### RIPER Workflow
- **Research**: Investigación inicial
- **Innovate**: Innovación y diseño
- **Plan**: Planificación detallada
- **Execute**: Ejecución
- **Review**: Revisión con consolidación eficiente de contexto

#### ContextKit
- Planificación en 4 fases
- Agentes de calidad especializados
- Código production-ready

### 18.2 Desarrollo Políglota
- Soporte para múltiples lenguajes de programación
- Optimizaciones específicas por lenguaje
- Respeto de convenciones idiomáticas

### 18.3 Migración de Frameworks
- Asistencia con actualizaciones de versión (ej: React 16 a 18)
- Estrategias de migración guiadas
- Preservación de APIs existentes

## 19. Colaboración en Equipo

### 19.1 Configuración Compartida
- Sincronización central de configuración
- Aplicación de guías de estilo del equipo
- Sesiones colaborativas de IA

### 19.2 Gestión de Sesiones
- Archivos de conversación `.jsonl`
- Visualizadores de historial de sesión
- Análisis de sesión con reportes HTML

### 19.3 Integración con Herramientas de Equipo
- Webhooks de GitHub para PRs e issues
- Integración con sistemas de tickets (Jira)
- Notificaciones de equipo

## 20. Monitoreo y Analytics

### 20.1 Herramientas de Monitoreo
- Statusline con barras de progreso
- Estadísticas persistentes (basadas en SQLite)
- Análisis de sesión con reportes HTML
- Monitoreo de uso de contexto
- Cálculo de burn rate

### 20.2 Reportes y Visualización
- Generación de reportes HTML
- Presentación UI en terminal con visualización de progreso
- Temas personalizables (dark/light, cumplimiento NO_COLOR)

### 20.3 Métricas Disponibles
- Tokens utilizados
- Comandos ejecutados
- Archivos modificados
- Tiempo de sesión
- Eficiencia de tareas

## 21. Personalización y Extensibilidad

### 21.1 Sistema de Plugins
- Plugins extensibles que agregan comandos personalizados
- Creación de agentes personalizados
- Integración con herramientas propietarias
- Instalación con comando `/plugin`
- Colecciones de slash commands, agentes, servidores MCP y hooks
- Funciona en terminal y VS Code
- Beta público para todos los usuarios de Claude Code
- Marketplaces de plugins disponibles

### 21.2 Comandos Slash Personalizados
- Creación de comandos específicos del equipo
- Codificación de procesos repetibles
- Meta-comandos para generar nuevos comandos

### 21.3 Plantillas y Snippets
- Plantillas específicas del proyecto
- Generación de componentes para frameworks
- Snippets reutilizables

## 22. Casos de Uso Específicos

### 22.1 Desarrollo Web
- Generación de componentes React/Vue/Angular
- Configuración de build tools (Webpack, Vite)
- Optimización de performance frontend
- Integración con APIs REST/GraphQL

### 22.2 Backend Development
- Creación de endpoints API
- Configuración de bases de datos
- Implementación de autenticación
- Manejo de middleware

### 22.3 DevOps y Infraestructura
- Generación de Dockerfiles
- Configuración de Kubernetes
- Scripts de deployment
- Configuración de CI/CD

### 22.4 Data Science y ML
- Preparación de datos
- Generación de modelos
- Análisis exploratorio
- Visualización de resultados

## 23. Integración con Voz

### 23.1 Soporte de Conversación Natural
- Integración con servicios de voz compatibles con OpenAI
- Instalación de servicio de voz open-source gratuito
- Whisper.cpp para transcripción
- Kokoro-FastAPI para síntesis

### 23.2 Comandos de Voz
- Activación por voz de comandos
- Dictado de código
- Navegación por voz

## 24. Gestión de Errores y Recuperación

### 24.1 Manejo de Interrupciones
- **Escape**: Interrumpir Claude durante cualquier fase
- **Doble Escape**: Saltar hacia atrás en el historial
- Preservación de contexto al interrumpir
- Redirección o expansión de instrucciones

### 24.2 Estrategias de Recuperación
- Modo dry-run para vista previa
- Lógica de reintentos con backoff exponencial
- Estrategias de fallback para tareas complejas
- Checkpointing automático

### 24.3 Debugging
- Logs detallados de operaciones
- Trazabilidad de decisiones de IA
- Modo verbose para diagnóstico

## 25. Mejores Prácticas Recomendadas

### 25.1 Control de Versiones
- Siempre trabajar en ramas feature
- Validar código generado antes del deployment
- Usar pre-commit hooks para revisión automatizada

### 25.2 Enfoques Incrementales
- Para refactoring a gran escala, usar procesamiento basado en checkpoints
- Modificaciones conscientes de dependencias
- Preservar APIs existentes

### 25.3 Seguridad
- Revisar todo código generado por seguridad
- No incluir credenciales en prompts
- Usar modo local cuando sea posible
- Mantener audit logs

### 25.4 Optimización de Contexto
- Limpiar contexto regularmente
- Usar sub-agentes para tareas específicas
- Aprovechar memoria branch-aware

## 26. Roadmap y Futuras Características

### 26.1 Próximamente
- Ventana de contexto de 1 millón de tokens
- Más integraciones MCP
- Mejoras en performance
- Nuevos modelos de IA

### 26.2 En Desarrollo
- Claude Code on the Web (beta actual)
- Delegación de tareas desde el navegador
- Infraestructura cloud gestionada por Anthropic
- Desarrollo paralelo de múltiples tareas

## 27. Recursos y Comunidad

### 27.1 Documentación Oficial
- docs.anthropic.com/en/docs/claude-code/overview
- GitHub: anthropics/claude-code
- Blog de Anthropic con actualizaciones

### 27.2 Comunidad
- Discord de Claude Developers
- GitHub Issues para reportes
- Awesome Claude Code repository
- Ejemplos y plantillas compartidas

### 27.3 Soporte
- Comando `/bug` interno
- Foro de la comunidad
- Documentación de troubleshooting

### 27.4 Herramientas de la Comunidad
- **ClaudeKit**: Toolkit CLI con auto-save checkpointing, hooks de calidad de código, generación y ejecución de especificaciones, y 20+ sub-agentes especializados (MIT License)
- **ClaudeLog**: Documentación, guías, tutoriales y mejores prácticas
- **Awesome Claude Code**: Lista curada de comandos, archivos y flujos de trabajo

## 28. Licencias y Términos

### 28.1 Modelo de Uso
- Cuenta Claude existente o facturación basada en uso de API
- Términos comerciales de servicio de Anthropic
- Política de privacidad de Anthropic

### 28.2 Open Source
- Repositorio público con 39.8k estrellas
- Contribuciones de la comunidad bienvenidas
- Licencia especificada en el repositorio

## 29. Claude Code SDK

### 29.1 Enfoque del SDK
- Búsqueda agentiva bajo demanda en lugar de pre-indexación
- Usa comandos grep, find, y glob como un desarrollador humano
- Enfoque radicalmente diferente de asistentes tradicionales
- Framework para construir agentes de IA con uso de herramientas

### 29.2 Características del SDK
- Integración con MCP
- Comportamientos personalizados más allá de tareas de codificación
- Gestión de sesiones para interacciones con estado
- Opciones de streaming y modo único

### 29.3 Migración a Claude Agent SDK
- Anteriormente conocido como Claude Code SDK
- Aislamiento mejorado intencional en v0.1.0
- El system prompt ya no tiene valores predeterminados de Claude Code
- Requiere configuración explícita para compatibilidad hacia atrás

## 30. Stack Técnico

### 30.1 Lenguajes
- **TypeScript**: 38.7% del código base
- **Python**: 28.5% del código base
- **Scripts**: PowerShell, Shell, Dockerfile

### 30.2 Modelos de IA
- Claude Sonnet 4.5 (modelo principal actual)
- Claude Opus 4.1 (modelo avanzado)
- Los mismos modelos que usan investigadores e ingenieros de Anthropic

---

*Última actualización: Octubre 2025*
*Fuente: Documentación oficial de Claude Code, repositorios GitHub, y recursos de la comunidad*