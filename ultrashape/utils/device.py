# Device utilities for cross-platform support (CUDA, MPS, CPU)

import torch

def get_device():
    """
    Get the best available device for the current platform.
    Returns: torch.device - cuda, mps, or cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_type():
    """
    Get the device type string for autocast.
    Returns: str - 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_autocast_dtype():
    """
    Get the appropriate dtype for autocast on the current platform.
    MPS doesn't support bfloat16, so we use float16 instead.
    Returns: torch.dtype
    """
    if torch.cuda.is_available():
        return torch.bfloat16
    elif torch.backends.mps.is_available():
        return torch.float16  # MPS doesn't support bfloat16
    else:
        return torch.float32  # CPU doesn't benefit from mixed precision

def empty_cache():
    """
    Clear the device cache (CUDA or MPS).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def is_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def is_mps():
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available()
