import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'OversampleQA'
copyright = '2024, Diogo Ribeiro'
author = 'Diogo Ribeiro'
version = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    # api.rst and api/oversampleqa.*.rst used to be excluded here: they were
    # orphaned autosummary stubs duplicating the hand-written api/<module>.rst
    # pages, kept out of the build by this pattern rather than deleted. They are
    # gone now, so the pattern is too.
    'gallery/*.ipynb',
    'gallery_examples/GALLERY_HEADER.rst',
]

try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except Exception:
    html_theme = 'alabaster'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Autosummary settings
autosummary_generate = False

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': ['gallery_examples'],
    'gallery_dirs': ['gallery'],
    'filename_pattern': r'.*\\.py$',
    'ignore_pattern': r'__init__',
    'download_all_examples': False,
    'plot_gallery': True,
    'abort_on_example_error': False,
    'remove_config_comments': True,
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build
