"""
Hardware detection service for optimal model loading
"""
import subprocess
import platform
import logging
import psutil
from typing import Dict

from app.config import settings

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect GPU and CPU capabilities for optimal model loading"""
    
    @staticmethod
    def has_nvidia_gpu() -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"NVIDIA GPU check failed: {e}")
            return False
    
    @staticmethod
    def has_amd_gpu() -> bool:
        """Check if AMD GPU is available"""
        try:
            result = subprocess.run(
                ['rocm-smi'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"AMD GPU check failed: {e}")
            return False
    
    @staticmethod
    def has_metal_support() -> bool:
        """Check if macOS Metal support is available"""
        if platform.system() != "Darwin":
            return False
        try:
            result = subprocess.run(
                ['sysctl', 'hw.model'],
                capture_output=True,
                timeout=5,
                text=True
            )
            # All Apple Silicon (M1, M2, M3, etc.) supports Metal
            model_info = result.stdout.lower()
            return 'apple' in model_info or any(
                f'm{i}' in model_info for i in range(1, 10)
            )
        except Exception as e:
            logger.debug(f"Metal support check failed: {e}")
            return False
    
    @staticmethod
    def get_available_memory() -> int:
        """Get available system memory in GB"""
        return int(psutil.virtual_memory().available / (1024**3))
    
    @staticmethod
    def get_total_memory() -> int:
        """Get total system memory in GB"""
        return int(psutil.virtual_memory().total / (1024**3))
    
    @staticmethod
    def has_gpu() -> bool:
        """Check if any GPU is available"""
        # Respect FORCE_CPU setting
        if settings.FORCE_CPU:
            logger.info("GPU disabled: FORCE_CPU=true")
            return False
        
        # Check ENABLE_GPU setting
        enable_gpu = settings.ENABLE_GPU.lower()
        if enable_gpu == "false":
            logger.info("GPU disabled: ENABLE_GPU=false")
            return False
        
        # Auto-detect
        has_any_gpu = (
            HardwareDetector.has_nvidia_gpu() or
            HardwareDetector.has_amd_gpu() or
            HardwareDetector.has_metal_support()
        )
        
        if has_any_gpu:
            logger.info("GPU detected and enabled")
        else:
            logger.info("No GPU detected, using CPU mode")
        
        return has_any_gpu
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get comprehensive system information for optimization"""
        has_gpu = HardwareDetector.has_gpu()
        
        info = {
            "ram_total_gb": HardwareDetector.get_total_memory(),
            "ram_available_gb": HardwareDetector.get_available_memory(),
            "has_gpu": has_gpu,
            "gpu_type": None,
            "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 4,
            "os": platform.system(),
            "processor": platform.processor()
        }
        
        # Detect GPU type
        if has_gpu:
            if HardwareDetector.has_nvidia_gpu():
                info["gpu_type"] = "NVIDIA"
            elif HardwareDetector.has_amd_gpu():
                info["gpu_type"] = "AMD"
            elif HardwareDetector.has_metal_support():
                info["gpu_type"] = "Apple Metal"
        
        return info
    
    @staticmethod
    def get_optimal_model() -> str:
        """
        Get optimal model based on hardware.
        Returns:
            Model name to use (e.g., 'gemma3:4b' for CPU)
        """
        has_gpu = HardwareDetector.has_gpu()
        memory_gb = HardwareDetector.get_available_memory()
        
        logger.info(f"Hardware: GPU={has_gpu}, Memory={memory_gb}GB")
        
        if has_gpu:
            # With GPU, use larger unquantized model
            if memory_gb >= 16:
                return "gemma3:4b"
            else:
                return "gemma3:4b"
        
        # CPU only - use quantized small model
        if memory_gb >= 16:
            return "gemma3:4b"
        elif memory_gb >= 8:
            return "gemma3:4b"
        else:
            return "gemma3:4b"
    
    @staticmethod
    def get_ollama_options() -> Dict:
        """Get optimized Ollama inference options"""
        has_gpu = HardwareDetector.has_gpu()
        cpu_cores = psutil.cpu_count(logical=False) or 4
        
        base_options = {
            "temperature": 0.3,
            "top_p": 0.9,
        }
        
        if has_gpu:
            # GPU can handle more tokens
            base_options.update({
                "num_predict": 1000,
            })
        else:
            # CPU - reduce tokens for speed
            base_options.update({
                "num_predict": 500,
                "num_thread": max(1, cpu_cores - 1),  # Use all but one core
            })
        
        return base_options
    
    @staticmethod
    def get_embedding_device() -> str:
        """Get device for embedding model (cuda, mps, or cpu)"""
        if settings.FORCE_CPU:
            return "cpu"
        
        if HardwareDetector.has_nvidia_gpu():
            return "cuda"
        elif HardwareDetector.has_metal_support():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def get_batch_size() -> int:
        """Get optimal batch size based on available memory"""
        memory_gb = HardwareDetector.get_available_memory()
        has_gpu = HardwareDetector.has_gpu()
        
        if has_gpu:
            if memory_gb >= 16:
                return 64
            elif memory_gb >= 8:
                return 32
            else:
                return 16
        else:
            # CPU mode - smaller batches
            if memory_gb >= 16:
                return 32
            elif memory_gb >= 8:
                return 16
            else:
                return 8


# Initialize hardware info at startup
system_info = HardwareDetector.get_system_info()
logger.info(f"System Info: {system_info}")