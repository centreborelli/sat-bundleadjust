import ast
import os
import re
import subprocess
from codecs import open

from setuptools import setup
from setuptools.command import develop, build_py

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


class CustomDevelop(develop.develop, object):
    """
    Class needed for "pip install -e ."
    """
    def run(self):
        subprocess.check_call("make", shell=True)
        super(CustomDevelop, self).run()


class CustomBuildPy(build_py.build_py, object):
    """
    Class needed for "pip install bundle_adjust"
    """
    def run(self):
        super(CustomBuildPy, self).run()
        subprocess.check_call("make", shell=True)
        subprocess.check_call("cp -r lib build/lib/", shell=True)


try:
    from wheel.bdist_wheel import bdist_wheel
    class BdistWheel(bdist_wheel):
        """
        Class needed to build platform dependent binary wheels
        """
        def finalize_options(self):
            bdist_wheel.finalize_options(self)
            self.root_is_pure = False

except ImportError:
    BdistWheel = None


extras_require = {"test": ["pytest"]}

setup(
    name=package_name,
    version=version,
    description="Register RPCs",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="",
    packages=["bundle_adjust", "bundle_adjust.feature_tracks"],
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass={'develop': CustomDevelop,
        'build_py': CustomBuildPy,
        'bdist_wheel': BdistWheel},
    python_requires=">=3.6",
    entry_points="""
              [console_scripts]
              bundle_adjust=bundle_adjust.cli:main
          """,
)
