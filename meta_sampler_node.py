"""
Implementación del Meta-Sampler Guiado (Seed-WAN)
Un sampler personalizado que orquesta WAN 2.2 y SeedVR2 para outpainting de video
"""
import torch
import comfy.samplers
import comfy.model_management
import comfy.sample
import comfy.utils
import latent_preview
from comfy.k_diffusion import sampling as k_diffusion_sampling

class MetaSamplerGuiado:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_original": ("LATENT",),
                "video_latente_guia": ("LATENT",),
                "mascara": ("MASK",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "VAE_WAN": ("VAE",),
                "feather_mask_pixels": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_video_outpainting",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, positive, negative, video_original, video_latente_guia, mascara, sampler_name, scheduler, denoise=1.0, VAE_WAN=None, feather_mask_pixels=5):
        # Implementación completa del Meta-Sampler Guiado
        device = comfy.model_management.get_torch_device()
        
        # Validar que los latentes tengan dimensiones compatibles
        original_samples = video_original["samples"]
        guia_samples = video_latente_guia["samples"]
        
        if original_samples.shape != guia_samples.shape:
            # Si las formas no coinciden, intentar adaptarlas
            if original_samples.shape[2:] != guia_samples.shape[2:]:
                # Escalar el latente guía para que coincida con el original
                guia_samples = torch.nn.functional.interpolate(
                    guia_samples,
                    size=original_samples.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                video_latente_guia = {"samples": guia_samples}
        
        # Crear el lienzo inicial combinando el video original con el latente guía
        # donde la máscara indica la región de outpainting
        lienzo_inicial = self.crear_lienzo_inicial(video_original, video_latente_guia, mascara)
        
        # Ejecutar el proceso de muestreo híbrido
        resultado = self.muestreo_hibrido(
            model, 
            lienzo_inicial, 
            positive, 
            negative, 
            cfg, 
            sampler_name,
            scheduler,
            steps,
            seed,
            denoise,
            video_latente_guia, 
            mascara,
            feather_mask_pixels
        )
        
        return (resultado,)

    def crear_lienzo_inicial(self, video_original, video_latente_guia, mascara):
        """
        Crea el lienzo inicial combinando el video original con el latente guía
        donde la máscara indica la región de outpainting
        """
        # Aseguramos que todos los latentes tengan la misma forma
        original = video_original["samples"]
        guia = video_latente_guia["samples"]
        mask = mascara
        
        # Expandir la máscara si es necesario para que coincida con las dimensiones del latente
        if len(mask.shape) < len(original.shape):
            # Añadir dimensiones para batch y canal si es necesario
            while len(mask.shape) < len(original.shape):
                mask = mask.unsqueeze(0)
        
        # Asegurar que la máscara tenga las mismas dimensiones espaciales que el latente
        if mask.shape[-2:] != original.shape[-2:]:
            # Interpolar la máscara para que coincida con el tamaño del latente
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), 
                size=original.shape[-2:], 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Asegurar que la máscara tenga los mismos batches que el original
        if mask.shape[0] != original.shape[0]:
            # Repetir la máscara para todos los frames si es necesario
            mask = mask.expand(original.shape[0], -1, -1)
        
        # Aplicar la máscara para combinar original y guía
        # Donde mask=0 se usa el original, donde mask=1 se usa la guía
        # Nota: en el contexto de outpainting, queremos preservar el centro (original)
        # y usar la guía en las áreas que serán restauradas
        lienzo = (1 - mask.unsqueeze(1)) * original + mask.unsqueeze(1) * guia
        
        return {"samples": lienzo}

    def muestreo_hibrido(self, model, lienzo_inicial, positive, negative, cfg, sampler_name, scheduler, steps, seed, denoise, video_latente_guia, mascara, feather_mask_pixels):
        """
        Implementa el proceso de muestreo híbrido que inyecta el latente guía
        en cada paso del proceso de denoising de WAN 2.2
        """
        device = comfy.model_management.get_torch_device()
        
        # Preparar la máscara latente con feathering para suavizado
        mascara_latente = self.preparar_mascara_latente(mascara, lienzo_inicial["samples"], feather_mask_pixels)
        
        # Función de inyección latente para cada paso de muestreo
        def pre_cfg_fn(args):
            """
            Función que se ejecuta antes de aplicar el modelo
            Inyecta el latente guía en el lienzo actual según la máscara
            """
            denoised = args["denoised"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            model = args["model"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            
            # Obtener el timestep actual a partir de sigma
            timestep = model.inner_model.inner_model.model_sampling.timestep(sigma)
            
            # Añadir ruido al latente guía para que coincida con el nivel de ruido del paso actual
            guia_ruidosa = self.agregar_ruido_a_guia(video_latente_guia["samples"], timestep, model, sigma)
            
            # Inyectar el latente guía en el lienzo actual según la máscara
            if mascara_latente.shape[0] != denoised.shape[0]:
                # Asegurar que la máscara tenga la misma cantidad de batches que los latentes
                mask_batch = mascara_latente.expand(denoised.shape[0], -1, -1)
            else:
                mask_batch = mascara_latente
            
            # Aplicar la máscara para mezclar la predicción original con la guía
            x_modificado = mask_batch.unsqueeze(1) * guia_ruidosa + (1 - mask_batch.unsqueeze(1)) * denoised
            
            args["denoised"] = x_modificado
            return args
            
        # Agregar el hook de pre_cfg para inyectar el latente guía
        model_options = model.model_options.copy()
        if "sampler_pre_cfg_function" in model_options:
            prev_pre_cfg = model_options["sampler_pre_cfg_function"]
            def new_pre_cfg(args):
                args = prev_pre_cfg(args)
                return pre_cfg_fn(args)
            model_options["sampler_pre_cfg_function"] = new_pre_cfg
        else:
            model_options["sampler_pre_cfg_function"] = pre_cfg_fn
            
        # Crear una copia del modelo con las opciones modificadas
        m = model.clone()
        m.model_options = model_options
        
        # Ejecutar el muestreo con las modificaciones
        resultado = comfy.sample.sample(
            m,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            lienzo_inicial["samples"],
            denoise=denoise
        )
        
        return {"samples": resultado}

    def preparar_mascara_latente(self, mascara, lienzo_latente, feather_mask_pixels=5):
        """
        Prepara la máscara latente con feathering (difuminado) para evitar artefactos de borde
        """
        mask = mascara
        
        # Expandir la máscara si es necesario
        if len(mask.shape) < len(lienzo_latente.shape):
            while len(mask.shape) < len(lienzo_latente.shape):
                mask = mask.unsqueeze(0)
        
        # Asegurar que la máscara tenga las mismas dimensiones espaciales que el latente
        if mask.shape[-2:] != lienzo_latente.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), 
                size=lienzo_latente.shape[-2:], 
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Asegurar que la máscara tenga los mismos batches que el latente
        if mask.shape[0] != lienzo_latente.shape[0]:
            mask = mask.expand(lienzo_latente.shape[0], -1, -1)
        
        # Aplicar feathering (difuminado) a la máscara para una transición suave
        if feather_mask_pixels > 0:
            import torchvision.transforms as transforms
            # El tamaño del kernel debe ser impar y proporcional al tamaño del feather
            kernel_size = feather_mask_pixels * 2 + 1
            sigma = feather_mask_pixels / 3.0
            transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            
            # Aplicar blur a cada frame de la máscara
            mascara_procesada = torch.zeros_like(mask)
            for i in range(mask.shape[0]):
                mascara_procesada[i] = transform(mask[i].unsqueeze(0)).squeeze(0)
            
            # Asegurar valores en el rango [0, 1]
            mascara_procesada = torch.clamp(mascara_procesada, 0, 1)
        else:
            mascara_procesada = mask
        
        return mascara_procesada

    def agregar_ruido_a_guia(self, latente_guia, timestep, model, sigma):
        """
        Agrega la cantidad apropiada de ruido al latente guía para que coincida
        con el nivel de ruido del paso actual del muestreo
        """
        import numpy as np
        
        # Implementación para agregar ruido al latente guía basado en el timestep
        try:
            # Obtener el sampling del modelo para acceder a la lógica específica del scheduler
            sampling = model.inner_model.inner_model.model_sampling
            
            # Calcular la fracción de ruido basado en el timestep
            # Convertir timestep a alfa_bar si es necesario
            if hasattr(sampling, 'sigmas'):
                # Encontrar el índice del sigma actual en la lista de sigmas
                sigma_current = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma if isinstance(sigma, (int, float)) else sigma.item()
                # Buscar sigma más cercano en la lista de sigmas
                sigmas = sampling.sigmas.cpu().numpy()
                idx_closest = (np.abs(sigmas - sigma_current)).argmin()
                
                # Calcular la proporción de avance en el proceso de muestreo
                total_steps = len(sigmas) - 1
                step_idx = idx_closest
                progress = step_idx / total_steps if total_steps > 0 else 0
                
            else:
                # Alternativa: usar directamente el timestep si está disponible
                timestep_val = timestep[0].item() if isinstance(timestep, torch.Tensor) and timestep.numel() > 0 else timestep
                progress = timestep_val / 999.0  # Asumiendo un rango típico de timesteps de 0-999
            
            # Generar ruido del mismo tamaño que el latente guía
            noise = torch.randn_like(latente_guia, device=latente_guia.device)
            
            # Aplicar ruido progresivamente basado en el avance del muestreo
            latente_guia_ruidoso = latente_guia * (1 - progress) + noise * progress
            
            return latente_guia_ruidoso
        except:
            # En caso de error, devolver el latente guía original
            return latente_guia