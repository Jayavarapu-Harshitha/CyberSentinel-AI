from importlib import import_module
MODULES = {'logistic_regression':'logistic_regression','random_forest':'random_forest','mlp':'mlp','hist_gradient_boosting':'hist_gradient_boosting','xgboost':'xgboost','lightgbm':'lightgbm'}
def build(name,seed=42):
    if name not in MODULES: raise ValueError(f'Unknown model: {name}')
    return import_module('cybersentinel.models.'+MODULES[name]).build(seed)
