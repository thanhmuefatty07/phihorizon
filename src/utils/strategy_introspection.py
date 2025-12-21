"""
Automatic strategy interface detection and code generation
No manual documentation needed
"""

import ast
import inspect
from typing import Type, Dict, List
from dataclasses import dataclass

@dataclass
class StrategyInterface:
    """Detected strategy interface"""
    class_name: str
    base_class: str
    methods: Dict[str, str]  # method_name -> signature
    required_params: List[str]

def detect_strategy_interface(strategy_class: Type) -> StrategyInterface:
    """
    Auto-detect strategy interface via AST analysis

    Returns:
        StrategyInterface with all methods and signatures
    """
    source = inspect.getsource(strategy_class)
    tree = ast.parse(source)

    # Find class definition
    class_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == strategy_class.__name__:
            class_def = node
            break

    if not class_def:
        raise ValueError(f"Could not find class definition for {strategy_class}")

    # Extract base class
    base_class = class_def.bases[0].id if class_def.bases else "ABC"

    # Extract all methods
    methods = {}
    for item in class_def.body:
        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
            # Get method signature
            sig = inspect.signature(getattr(strategy_class, item.name))
            methods[item.name] = str(sig)

    # Extract required parameters from __init__
    required_params = []
    for item in class_def.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            for arg in item.args.args[1:]:  # Skip 'self'
                required_params.append(arg.arg)

    return StrategyInterface(
        class_name=strategy_class.__name__,
        base_class=base_class,
        methods=methods,
        required_params=required_params
    )

def generate_strategy_wrapper(interface: StrategyInterface) -> str:
    """
    Generate correct integration code automatically

    Returns:
        Python code as string (ready to execute)
    """
    code = f"""
# Auto-generated strategy wrapper for {interface.class_name}
# Base: {interface.base_class}

from src.strategies.{interface.class_name.lower()} import {interface.class_name}

def create_{interface.class_name.lower()}(params: dict):
    '''
    Auto-generated factory function

    Required params: {interface.required_params}
    '''
    required = {interface.required_params}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing parameters: {{missing}}")

    return {interface.class_name}(params)

def execute_{interface.class_name.lower()}(strategy, data):
    '''
    Auto-generated execution function

    Available methods:
"""

    for method_name, signature in interface.methods.items():
        code += f"    - {method_name}{signature}\n"

    code += f"""    '''
    # Auto-detect correct method
    if hasattr(strategy, 'generate_signals'):
        return strategy.generate_signals(data)
    elif hasattr(strategy, 'get_indicators'):
        return strategy.get_indicators()
    else:
        raise AttributeError(
            f"Strategy {{type(strategy).__name__}} has no recognized interface"
        )
"""

    return code