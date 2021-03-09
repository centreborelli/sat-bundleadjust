import ast
import os
import re
from codecs import open

from setuptools import setup

_version_re = re.compile(r"__version__\s+=\s+(.*)")
package_name = "bundle_adjust"

with open(os.path.join(package_name, "__init__.py"), "rb") as f:
    version = str(
        ast.literal_eval(_version_re.search(f.read().decode("utf-8")).group(1))
    )

with open("requirements.txt", "r", "utf-8") as f:
    install_requires = f.readlines()

with open("README.md", "r", "utf-8") as f:
    readme = f.read()

extras_require = {"test": ["pytest"]}

setup(
    name=package_name,
    version=version,
    description="Register RPCs",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="a.lahlou-mimi@kayrros.com",
    packages=["bundle_adjust", "bundle_adjust.feature_tracks"],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    entry_points="""
              [console_scripts]
              bundle_adjust=bundle_adjust.cli:main
          """
)
