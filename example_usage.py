import torch
import folder_paths
import comfy.model_management
from .meta_sampler_node import MetaSamplerGuiado

# Este archivo proporciona un ejemplo de cómo se integraría el nodo en ComfyUI
# El nodo ya está completamente funcional con todas las conexiones y lógica implementada

"""
Ejemplo de uso del Meta-Sampler Guiado:

1. Carga tu modelo WAN 2.2
2. Prepara tu video original como latente
3. Procesa el video original con SeedVR2 para obtener el latente guía
4. Crea una máscara para indicar la región de outpainting
5. Conecta todos los inputs al nodo MetaSamplerGuiado
6. Ejecuta el muestreo para obtener el video de outpainting con coherencia temporal
"""