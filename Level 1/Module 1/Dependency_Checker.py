"""
requirements.txt
       ↓
Parse requirement
       ↓
importlib → Is package installed?
       ↓
   ┌───┴────┐
   NO       YES
   ↓         ↓
Missing   Get installed version
             ↓
      Does it satisfy requirement?
          ↙          ↘
        NO            YES
        ↓              ↓
  Mismatch       Check latest PyPI
                       ↓
                Installed < Latest?
                  ↙          ↘
                YES          NO
                 ↓            ↓
              Outdated    Up to date"""

import importlib.util
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.version import Version

def read_requirements(filename):  # Read package requirements
    packages = []

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            requirement = Requirement(line)
            packages.append(requirement)

    return packages

def get_installed_version(package_name):  # Get installed package version
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        capture_output=True,
        text=True
    )

    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()

    return None

def check_package(requirement):  # Check package status
    package_name = requirement.name
    import_name = package_name.replace("-", "_")

    if importlib.util.find_spec(import_name) is None:
        return "Missing", None, None

    installed_version = get_installed_version(package_name)

    if installed_version is None:
        return "Missing", None, None

    # Check whether installed version satisfies requirements.txt
    if Version(installed_version) not in requirement.specifier:
        return "Requirement mismatch", installed_version, None

    # Check latest version available from PyPI
    result = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", package_name],
        capture_output=True,
        text=True
    )

    latest_version = None

    for line in result.stdout.splitlines():
        if "Available versions:" in line:
            versions = line.split(":", 1)[1].strip().split(",")
            if versions:
                latest_version = versions[0].strip()
            break

    if latest_version and Version(installed_version) < Version(latest_version):
        return "Outdated", installed_version, latest_version

    return "Up to date", installed_version, latest_version

def main():  # Run dependency checker
    requirements = read_requirements("requirements.txt")

    print("Dependency Checker")
    print("-" * 70)

    for requirement in requirements:
        status, installed, latest = check_package(requirement)

        print(f"\n{requirement.name}: {status}")

        if status == "Missing":
            print(f"  Required: {requirement}")

        elif status == "Requirement mismatch":
            print(f"  Required: {requirement}")
            print(f"  Installed: {installed}")

        elif status == "Outdated":
            print(f"  Installed: {installed}")
            print(f"  Latest: {latest}")

        else:
            print(f"  Installed: {installed}")

if __name__ == "__main__":
    main()