# Meta-Sampler Guiado (Seed-WAN) para ComfyUI

## Descripción del Proyecto
Este proyecto implementa el "Meta-Sampler Guiado" (también conocido como Seed-WAN), un nodo personalizado para ComfyUI que implementa una técnica innovadora de muestreo híbrido. La implementación fusiona un modelo generativo de video (como WAN 2.2) con un modelo de restauración y súper-resolución (como SeedVR2) a nivel de espacio latente.

## Características Principales
- **Fusión de procesos de muestreo**: En lugar de un pipeline secuencial (generar y luego restaurar), el nodo utiliza la salida de SeedVR2 como un "ancla latente" de alta fidelidad.
- **Inyección de latentes de restauración**: En cada paso del bucle de denoising del modelo generativo (WAN), se inyectan forzosamente latentes de guía pre-procesados en el área del contenido original.
- **Outpainting de video temporalmente coherente**: Obliga al modelo generativo a enfocar toda su inferencia creativa (guiada por prompts de texto) únicamente en la región de outpainting.
- **Transición de costura perfecta**: Asegura una coherencia temporal heredada del modelo de restauración.
- **Control total mediante prompts de texto**: Mantiene la capacidad de controlar el área generada con condiciones de texto.

## Componentes del Proyecto
- `__init__.py`: Archivo de inicialización del nodo personalizado
- `meta_sampler_node.py`: Implementación principal del Meta-Sampler Guiado
- `README.md`: Documentación del proyecto
- `pyproject.toml`: Archivo de configuración del proyecto
- `example_usage.py`: Ejemplo de uso del nodo
- `.gitignore`: Archivo para ignorar archivos temporales y sensibles
- `CHANGELOG.md`: Registro de cambios del proyecto
- `LICENSE`: Licencia MIT del proyecto

## Implementación Técnica
### Flujo de Proceso
1. **Fase 1 - Preparación**: El video original se procesa con SeedVR2 para crear un "ancla latente" nítida y temporalmente estable
2. **Fase 2 - Bucle de Muestreo Híbrido**: Inyección del latente guía en cada paso del muestreo de WAN 2.2
3. **Fase 3 - Decodificación**: Conversión final del latente al dominio de píxeles

### Soluciones Técnicas
- **Compatibilidad de Espacio Latente**: Manejo de incompatibilidades entre VAEs de diferentes modelos
- **Gestión de Costura**: Implementación de feathering para evitar artefactos de borde
- **Coherencia Temporal Cruzada**: Aprovechamiento de la lógica temporal de ambos modelos para mejorar la estabilidad

## Inputs del Nodo
- `model`: Modelo WAN 2.2
- `seed`: Semilla para la generación
- `steps`: Número de pasos de muestreo
- `cfg`: Escala del guía de clasificación
- `positive`: Conditioning positivo
- `negative`: Conditioning negativo
- `video_original`: Latente del video original
- `video_latente_guia`: Latente del video procesado con SeedVR2
- `mascara`: Máscara indicando la región de outpainting
- `sampler_name`: Nombre del sampler a usar
- `scheduler`: Scheduler para el muestreo
- `denoise`: Factor de ruido
- Inputs opcionales: `VAE_WAN`, `feather_mask_pixels`

## Output del Nodo
- `latent_video_outpainting`: Video latente con outpainting aplicado

## Requisitos
- ComfyUI
- Modelos WAN 2.2 y SeedVR2
- (Opcional) VAE compatible con WAN 2.2

## Instalación
1. Clonar o descargar el repositorio
2. Colocar la carpeta en el directorio `custom_nodes` de ComfyUI
3. Reiniciar ComfyUI
4. El nodo estará disponible en la categoría "sampling"

## Casos de Uso
- Outpainting de video con alta fidelidad y coherencia temporal
- Expansión de lienzo de videos manteniendo la calidad y consistencia
- Edición de videos donde se requiere generar contenido adicional que se integre perfectamente con el original
- Aplicaciones de post-producción y efectos visuales asistidos por IA

## Contribuciones
Las contribuciones son bienvenidas. Si encuentras problemas o tienes sugerencias para mejorar la implementación, por favor abre un issue o envía un pull request.

## Autor
Aishor (ax.rocholl@gmail.com)

## Licencia
MIT License - Ver el archivo LICENSE para más detalles.

## Repositorio GitHub
https://github.com/Aishor/comfyui-meta-sampler-guiado