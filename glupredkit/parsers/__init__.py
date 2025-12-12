import importlib


def safe_import(module_name, class_name, package="glupredkit.parsers"):
    """Attempts to import a module and return its class. If the module is missing, raises ImportError when accessed."""
    try:
        module = importlib.import_module(module_name, package=package)
        return getattr(module, class_name)
    except ModuleNotFoundError:
        return None  # Return None instead of raising an error


# Define parsers with safe import
AppleHealthParser = safe_import(".apple_health", "Parser", package="glupredkit.parsers")
NightscoutParser = safe_import(".nightscout", "Parser", package="glupredkit.parsers")
OhioT1DMParser = safe_import(".ohio_t1dm", "Parser", package="glupredkit.parsers")
OpenAPSParser = safe_import(".open_aps", "Parser", package="glupredkit.parsers")
ShanghaiT1DMParser = safe_import(".shanghai_t1dm", "Parser", package="glupredkit.parsers")
T1DexiParser = safe_import(".t1dexi", "Parser", package="glupredkit.parsers")
TidepoolAPIParser = safe_import(".tidepool", "Parser", package="glupredkit.parsers")
TidepoolDatasetParser = safe_import(".tidepool_dataset", "Parser", package="glupredkit.parsers")
IOBP2Parser = safe_import(".IOBP2", "Parser", package="glupredkit.parsers")
DiaTrendParser = safe_import(".diatrendparser", "Parser", package="glupredkit.parsers")

