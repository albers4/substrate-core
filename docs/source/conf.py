import os
import sys

sys.path.insert(0, os.path.abspath("../../substrate-core-py"))

project = "substrate"
copyright = ""
author = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

cores_path = ["_cores"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
