"""
Meta-Sampler Guiado (Seed-WAN) para ComfyUI
Implementación del sampler híbrido que fusiona WAN 2.2 con SeedVR2
"""
from .meta_sampler_node import MetaSamplerGuiado

NODE_CLASS_MAPPINGS = {
    "MetaSamplerGuiado": MetaSamplerGuiado
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetaSamplerGuiado": "Meta-Sampler Guiado (Seed-WAN)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']