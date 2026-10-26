from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Reads requirements.txt and returns list of dependencies."""
    requirements = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            req = line.strip()
            if req and not req.startswith("#") and req != HYPHEN_E_DOT:
                requirements.append(req)
    return requirements

setup(
    name="aiml_project",
    version="0.1.0",
    author="Data Science & ML Team",
    author_email="example@aiml.com",
    description="A modular, production-grade Artificial Intelligence & Machine Learning framework",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.9",
)
