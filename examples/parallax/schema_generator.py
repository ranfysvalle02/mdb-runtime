#!/usr/bin/env python3
"""
Dynamic Schema Generator for Parallax Lenses

Generates Pydantic models on-the-fly from database-stored schema definitions.
"""
import logging
from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

logger = logging.getLogger("Parallax.SchemaGenerator")


# Type mapping from string to Python types
TYPE_MAPPING = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "List[str]": List[str],
    "List[int]": List[int],
    "Literal": str,  # For Literal types, we'll handle them specially
}


def create_dynamic_model(
    model_name: str,
    fields: List[Dict[str, Any]]
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model from field definitions.
    
    Args:
        model_name: Name of the model class (e.g., "MarketingView")
        fields: List of field definitions, each with:
            - name: str (field name)
            - type: str (Python type as string)
            - description: str (field description)
            - default: Any (optional default value)
            - constraints: Dict (optional constraints like ge, le, min_length, max_length)
            - literal_values: List[str] (for Literal types)
    
    Returns:
        A Pydantic BaseModel class
    """
    model_fields: Dict[str, Any] = {}
    
    for field_def in fields:
        field_name = field_def.get("name")
        if not field_name:
            logger.warning(f"Skipping field without name: {field_def}")
            continue
        
        field_type_str = field_def.get("type", "str")
        description = field_def.get("description", "")
        default_value = field_def.get("default", ...)  # Use ... for required fields
        constraints = field_def.get("constraints", {})
        literal_values = field_def.get("literal_values", [])
        
        # Determine Python type
        if literal_values:
            # Create Literal type
            from typing import Literal
            python_type = Literal[tuple(literal_values)]
        elif field_type_str.startswith("List["):
            # Handle List types
            inner_type = field_type_str.replace("List[", "").replace("]", "").strip()
            if inner_type == "str":
                python_type = List[str]
            elif inner_type == "int":
                python_type = List[int]
            else:
                python_type = List[str]  # Default to List[str]
        else:
            python_type = TYPE_MAPPING.get(field_type_str, str)
        
        # Build Field with constraints
        field_kwargs = {"description": description}
        field_kwargs.update(constraints)
        
        # Set default if provided
        if default_value is not ...:
            field_kwargs["default"] = default_value
        
        model_fields[field_name] = (python_type, Field(**field_kwargs))
    
    # Create the model dynamically
    try:
        DynamicModel = create_model(model_name, **model_fields)
        logger.info(f"Created dynamic model: {model_name} with {len(fields)} fields")
        return DynamicModel
    except Exception as e:
        logger.error(f"Failed to create dynamic model {model_name}: {e}", exc_info=True)
        raise


def get_default_lens_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get default lens configurations.
    Focused on Relevance (personalized value) and Technical (practical engineering insights).
    """
    return {
        "Relevance": {
            "lens_name": "Relevance",
            "prompt_template": (
                "You are the Relevance Specialist. Analyze this news title in context of watchlist: {watchlist}\n\n"
                "Explain why this story matters given the watchlist. Be specific about watchlist connections. "
                "Keep responses concise (1-2 sentences per field). Do not hallucinate. Output valid JSON.\n\n"
                "{format_instructions}"
            ),
            "schema_fields": [
                {
                    "name": "relevance_score",
                    "type": "int",
                    "description": "Relevance to watchlist (1-100)",
                    "default": None,
                    "constraints": {"ge": 1, "le": 100}
                },
                {
                    "name": "why_it_matters",
                    "type": "str",
                    "description": "Why this matters (1 sentence)",
                    "default": None
                },
                {
                    "name": "key_insight",
                    "type": "str",
                    "description": "Most important takeaway (1 sentence)",
                    "default": None
                },
                {
                    "name": "urgency",
                    "type": "Literal",
                    "description": "Urgency level",
                    "default": None,
                    "literal_values": ["Low", "Medium", "High"],
                    "display_type": "badge"
                }
            ]
        },
        "Technical": {
            "lens_name": "Technical",
            "prompt_template": (
                "You are the Technical Specialist. Provide concise technical assessment: performance, complexity, readiness.\n\n"
                "Analyze this from an engineering perspective. Be concise (1-2 sentences per field). "
                "Focus on practical implications. Do not hallucinate. Output valid JSON.\n\n"
                "{format_instructions}"
            ),
            "schema_fields": [
                {
                    "name": "performance",
                    "type": "str",
                    "description": "Key performance characteristics (1 sentence)",
                    "default": None
                },
                {
                    "name": "complexity",
                    "type": "Literal",
                    "description": "Implementation complexity",
                    "default": None,
                    "literal_values": ["Low", "Medium", "High"],
                    "display_type": "badge"
                },
                {
                    "name": "readiness",
                    "type": "Literal",
                    "description": "Production readiness",
                    "default": None,
                    "literal_values": ["Experimental", "Beta", "Production"],
                    "display_type": "badge"
                },
                {
                    "name": "use_case",
                    "type": "str",
                    "description": "Best use case (1 sentence)",
                    "default": None
                }
            ]
        }
    }

