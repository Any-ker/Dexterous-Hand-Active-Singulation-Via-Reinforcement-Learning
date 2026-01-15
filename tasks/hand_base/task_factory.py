"""
Task Factory for creating and configuring tasks.
"""

from typing import Dict, Any, Optional, Type, List, Tuple
import importlib
import os


class TaskFactory:
    """Factory class for creating task instances."""
    
    _task_registry: Dict[str, Type] = {}
    
    @classmethod
    def register_task(cls, task_name: str, task_class: Type):
        """
        Register a task class.
        
        Args:
            task_name: Name identifier for the task
            task_class: Task class to register
        """
        cls._task_registry[task_name] = task_class
    
    @classmethod
    def create_task(
        self,
        task_name: str,
        cfg: Dict[str, Any],
        sim_params: Any,
        physics_engine: Any,
        device_type: str,
        device_id: int,
        headless: bool,
        **kwargs
    ) -> Any:
        """
        Create a task instance.
        
        Args:
            task_name: Name of the task to create
            cfg: Configuration dictionary
            sim_params: Simulation parameters
            physics_engine: Physics engine type
            device_type: Device type ("cuda" or "cpu")
            device_id: Device ID
            headless: Whether to run in headless mode
            **kwargs: Additional arguments for task initialization
            
        Returns:
            Task instance
        """
        if task_name not in self._task_registry:
            raise ValueError(
                f"Task '{task_name}' not registered. "
                f"Available tasks: {list(self._task_registry.keys())}"
            )
        
        task_class = self._task_registry[task_name]
        
        return task_class(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
            **kwargs
        )
    
    @classmethod
    def create_task_from_config(
        cls,
        config_path: str,
        task_name: Optional[str] = None,
        **override_kwargs
    ) -> Any:
        """
        Create a task from a configuration file.
        
        Args:
            config_path: Path to configuration file
            task_name: Optional task name override
            **override_kwargs: Configuration overrides
            
        Returns:
            Task instance
        """
        import yaml
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Apply overrides
        for key, value in override_kwargs.items():
            keys = key.split('.')
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        
        # Get task name from config or parameter
        if task_name is None:
            task_name = cfg.get('task', {}).get('name', 'DexGrasp')
        
        # This is a simplified version - in practice, you'd need to
        # properly set up sim_params, physics_engine, etc.
        # For now, this serves as a template
        raise NotImplementedError(
            "Full implementation requires sim_params and physics_engine setup. "
            "Use create_task() with proper parameters instead."
        )
    
    @classmethod
    def list_available_tasks(cls) -> List[str]:
        """List all registered task names."""
        return list(cls._task_registry.keys())
    
    @classmethod
    def load_task_from_module(cls, module_path: str, task_class_name: str, task_name: str):
        """
        Dynamically load and register a task class from a module.
        
        Args:
            module_path: Path to the module (e.g., "tasks.dex_grasp")
            task_class_name: Name of the task class
            task_name: Name to register the task under
        """
        try:
            module = importlib.import_module(module_path)
            task_class = getattr(module, task_class_name)
            cls.register_task(task_name, task_class)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load task from {module_path}.{task_class_name}: {e}")


class TaskConfigValidator:
    """Validates task configuration dictionaries."""
    
    REQUIRED_KEYS = {
        'env': ['numEnvs', 'numObservations', 'numActions', 'episodeLength'],
    }
    
    OPTIONAL_KEYS = {
        'env': [
            'numStates', 'controlFrequencyInv', 'envSpacing',
            'aggregateMode', 'enableDebugVis', 'headless'
        ],
    }
    
    @classmethod
    def validate(cls, cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            cfg: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required keys
        for section, keys in cls.REQUIRED_KEYS.items():
            if section not in cfg:
                errors.append(f"Missing required section: '{section}'")
                continue
            
            for key in keys:
                if key not in cfg[section]:
                    errors.append(f"Missing required key: '{section}.{key}'")
        
        # Validate value types and ranges
        if 'env' in cfg:
            env_cfg = cfg['env']
            
            # Validate numEnvs
            if 'numEnvs' in env_cfg:
                if not isinstance(env_cfg['numEnvs'], int) or env_cfg['numEnvs'] <= 0:
                    errors.append("'env.numEnvs' must be a positive integer")
            
            # Validate episodeLength
            if 'episodeLength' in env_cfg:
                if not isinstance(env_cfg['episodeLength'], (int, float)) or env_cfg['episodeLength'] <= 0:
                    errors.append("'env.episodeLength' must be a positive number")
            
            # Validate numObservations
            if 'numObservations' in env_cfg:
                if not isinstance(env_cfg['numObservations'], int) or env_cfg['numObservations'] <= 0:
                    errors.append("'env.numObservations' must be a positive integer")
            
            # Validate numActions
            if 'numActions' in env_cfg:
                if not isinstance(env_cfg['numActions'], int) or env_cfg['numActions'] <= 0:
                    errors.append("'env.numActions' must be a positive integer")
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_and_fix(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and attempt to fix configuration issues.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            Fixed configuration dictionary
        """
        cfg = cfg.copy()
        
        # Ensure required sections exist
        if 'env' not in cfg:
            cfg['env'] = {}
        
        # Set defaults for missing optional keys
        defaults = {
            'env': {
                'numStates': 0,
                'controlFrequencyInv': 1,
                'envSpacing': 2.0,
                'aggregateMode': 1,
                'enableDebugVis': False,
                'headless': True,
            }
        }
        
        for section, default_values in defaults.items():
            if section not in cfg:
                cfg[section] = {}
            for key, default_value in default_values.items():
                if key not in cfg[section]:
                    cfg[section][key] = default_value
        
        return cfg

