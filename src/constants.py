"""
This module contains the paths to the configuration and schema files. The paths are
normalized to ensure they are in the correct format for the current operating system.

The configuration file is expected to be in YAML format and contains various settings
for the application. The schema file is also expected to be in YAML format and defines
the structure of the data the application works with.
"""

from os.path import normpath

CONFIGS = normpath("conf/configs.yaml")
SCHEMA = normpath("conf/schema.yaml")
PARAMS = normpath("conf/params.yaml")
