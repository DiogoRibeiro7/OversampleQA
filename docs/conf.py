import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

import oversampleqa

project = 'OversampleQA'
author = 'Diogo Ribeiro'
release = oversampleqa.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
