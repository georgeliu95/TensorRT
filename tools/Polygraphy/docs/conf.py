import sys
import os
ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.pardir)
sys.path.insert(0, ROOT_DIR)
import polygraphy

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

# Want to be able to generate docs with no dependencies installed
autodoc_mock_imports = ["tensorrt", "onnx", "numpy", "tensorflow", "onnx_graphsurgeon", "onnxruntime", "onnxtf", "tf2onnx"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "exclude-members": "activate_impl, deactivate_impl, BaseNetworkFromOnnx, BaseDataLoader",
    "special-members": "__call__, __getitem__",
}

autodoc_member_order = "bysource"

autodoc_inherit_docstrings = True

autosummary_generate = True

source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Polygraphy'
copyright = '2020, NVIDIA'
author = 'NVIDIA'

version = polygraphy.__version__
# The full version, including alpha/beta/rc tags.
release = version

# Style
pygments_style = 'colorful'

html_theme = 'sphinx_rtd_theme'

# Use the TRT theme and NVIDIA logo
html_static_path = ['_static']

html_logo = '_static/img/nvlogo_white.png'

# Hide source link
html_show_sourcelink = False

# Output file base name for HTML help builder.
htmlhelp_basename = 'TensorRTdoc'

# For constructor arguments to show up in Sphinx generated doc
autoclass_content = 'both'

# Unlimited depth sidebar.
html_theme_options = {
    'navigation_depth': -1
}

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# Allows us to override the default page width in the Sphinx theme.
def setup(app):
    app.add_css_file('style.css')
