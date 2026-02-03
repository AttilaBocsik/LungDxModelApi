# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Nagyon fontos: a 'src' mappát kell megadni, hogy a 'dicom_labeler' látható legyen
sys.path.insert(0, os.path.abspath('../..'))

project = 'LungDxModelApi'
copyright = '2026, Attila Bocsik'
author = 'Attila Bocsik'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Kinyeri a docstringeket a kódból
    'sphinx.ext.napoleon', # Ez kezeli a Google-stílusú kommenteket
    'sphinx.ext.viewcode', # Linket tesz a forráskódhoz
    'sphinx_rtd_theme',    # A modern téma
    'myst_parser',         # README.md integrálása
]

exclude_patterns = []

language = 'hu'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Téma beállítása
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Ha a Sphinx nem találja a PyQt6-ot a gépén, add hozzá ezt:
autodoc_mock_imports = ["PyQt6", "requests", "pydicom", "xgboost", "pydantic_settings", "pydantic", "dask"]
