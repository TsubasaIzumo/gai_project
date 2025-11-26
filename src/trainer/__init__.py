from .generator import GeneratorModule

# Import PretrainedGeneratorModule
try:
    from .generator_pretrained import PretrainedGeneratorModule
except Exception as e:
    # Re-raise the exception with more context
    raise ImportError(
        f"Failed to import PretrainedGeneratorModule from generator_pretrained: {e}\n"
        "This might be due to:\n"
        "1. Missing 'diffusers' package - install with: pip install diffusers>=0.21.0\n"
        "2. Syntax error in generator_pretrained.py\n"
        "3. Missing dependencies for generator_pretrained.py"
    ) from e
