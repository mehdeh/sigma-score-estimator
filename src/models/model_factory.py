"""
Model factory for creating omega estimator models.
"""

from .resnet_omega_x import resnet18_omega_x
from .resnet_omega_xs import resnet18_omega_xs


def create_model(model_type, **kwargs):
    """
    Create a model based on the specified type.
    
    Args:
        model_type (str): Type of model to create. Options:
            - 'omega_x': Model that takes only image as input
            - 'omega_x_sigma': Model that takes image and sigma as input
        **kwargs: Additional keyword arguments to pass to the model constructor
    
    Returns:
        PyTorch model instance
    
    Raises:
        ValueError: If model_type is not recognized
    
    Example:
        >>> model = create_model('omega_x_sigma')
        >>> output = model(images, sigmas)
    """
    if model_type == 'omega_x':
        return resnet18_omega_x(**kwargs)
    elif model_type == 'omega_x_sigma':
        return resnet18_omega_xs(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Expected 'omega_x' or 'omega_x_sigma'."
        )


def get_available_models():
    """
    Get a list of available model types.
    
    Returns:
        List of model type strings
    """
    return ['omega_x', 'omega_x_sigma']

