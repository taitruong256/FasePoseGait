from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module
import logging

logger = logging.getLogger(__name__)

# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([str(package_dir)]):

    # import the module and iterate through its attributes
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        logger.warning("Skipping loss module %s because dependency is missing: %s", module_name, exc)
        continue
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute):
            # Add the class to this package's variables
            globals()[attribute_name] = attribute
