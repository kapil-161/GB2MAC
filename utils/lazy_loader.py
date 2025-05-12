"""Lazy module loading to reduce initial memory footprint and startup time."""
import importlib
import logging

logger = logging.getLogger(__name__)

class LazyLoader:
    """Lazily import modules only when needed."""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None
        logger.debug(f"Setting up lazy loading for {module_name}")
        
    def __getattr__(self, name):
        if self._module is None:
            # Simple version without logging
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)

# Pre-define commonly used lazy modules
LAZY_MODULES = {
    'pandas': LazyLoader('pandas'),
    'numpy': LazyLoader('numpy'),
    'plotly.graph_objects': LazyLoader('plotly.graph_objects'),
    'plotly.express': LazyLoader('plotly.express'),
    'dash_bootstrap_components': LazyLoader('dash_bootstrap_components'),
}

def get_lazy(module_name):
    """Get a lazily loaded module."""
    if module_name not in LAZY_MODULES:
        LAZY_MODULES[module_name] = LazyLoader(module_name)
    return LAZY_MODULES[module_name]