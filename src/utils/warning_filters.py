# src/utils/warning_filters.py

import warnings
import logging
from typing import List, Dict, Any


def configure_warnings_for_dpo_driver():
    """
    Configure warnings for DPO-Driver project to reduce noise while preserving important warnings.
    """
    
    # 1. Suppress MiniWoB++ environment registration warnings
    warnings.filterwarnings(
        "ignore", 
        category=UserWarning, 
        message=".*Overriding environment miniwob.*already in registry.*"
    )
    
    # 2. Suppress Selenium deprecation warnings that we can't control
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="selenium.*"
    )
    
    # 3. Suppress transformers tokenizer warnings for known issues
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Mismatch between tokenized prompt.*"
    )
    
    # 4. Suppress gym/gymnasium version warnings
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*WARN: Overriding environment.*"
    )
    
    # 5. Set up logging to capture filtered warnings in our logs if needed
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.ERROR)  # Only show error-level warnings
    
    print("[WARNING FILTER] Configured warning filters for cleaner output")


def reset_warnings():
    """Reset warnings to default state."""
    warnings.resetwarnings()
    print("[WARNING FILTER] Reset warnings to default state")


def get_warning_summary() -> Dict[str, Any]:
    """
    Get a summary of current warning configuration.
    
    Returns:
        Dict containing information about current warning filters
    """
    return {
        "filters_count": len(warnings.filters),
        "filters_active": [
            {
                "action": f.action,
                "category": f.category.__name__ if f.category else "Any",
                "message": f.message.pattern if hasattr(f.message, 'pattern') else str(f.message),
                "module": f.module.pattern if hasattr(f.module, 'pattern') else str(f.module)
            }
            for f in warnings.filters
        ]
    }


# Apply filters when this module is imported
configure_warnings_for_dpo_driver()