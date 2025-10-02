#!/usr/bin/env python3
"""
Build and deployment script for MEWS
Handles environment setup, testing, and deployment
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(command: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    try:
        logger.info(f"Running: {command}")
        result = subprocess.run(
            command, shell=True, cwd=cwd, check=True, capture_output=True, text=True
        )
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(e.stderr)
        return False


def setup_environment():
    """Set up Python virtual environment and install dependencies."""
    logger.info("Setting up environment...")

    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        if not run_command(f"{sys.executable} -m venv venv"):
            return False

    # Determine activation script based on OS
    if os.name == "nt":  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix-like
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"

    # Install dependencies
    commands = [
        f"{pip_command} install --upgrade pip",
        f"{pip_command} install -r requirements.txt",
    ]

    for command in commands:
        if not run_command(command):
            return False

    logger.info("Environment setup completed successfully!")
    return True


def setup_dev_environment():
    """Set up development environment with additional tools."""
    logger.info("Setting up development environment...")

    if not setup_environment():
        return False

    # Determine pip command based on OS
    pip_command = "venv\\Scripts\\pip" if os.name == "nt" else "venv/bin/pip"
    python_command = "venv\\Scripts\\python" if os.name == "nt" else "venv/bin/python"

    commands = [
        f"{pip_command} install -r requirements-dev.txt",
        f"{python_command} -c \"import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')\"",
        "pre-commit install",
    ]

    for command in commands:
        if not run_command(command):
            return False

    logger.info("Development environment setup completed!")
    return True


def run_tests():
    """Run the test suite."""
    logger.info("Running tests...")

    # Determine python command based on OS
    python_command = "venv\\Scripts\\python" if os.name == "nt" else "venv/bin/python"

    commands = [
        f"{python_command} -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term",
    ]

    for command in commands:
        if not run_command(command):
            return False

    logger.info("Tests completed successfully!")
    return True


def run_linting():
    """Run code quality checks."""
    logger.info("Running code quality checks...")

    commands = [
        "flake8 src/ main.py streamlit_app.py --max-line-length=88 --extend-ignore=E203,W503",
        "black --check src/ main.py streamlit_app.py",
        "isort --check-only src/ main.py streamlit_app.py",
        "mypy src/ --ignore-missing-imports",
        "bandit -r src/ -f json -o bandit-report.json",
        "safety check",
    ]

    all_passed = True
    for command in commands:
        if not run_command(command):
            all_passed = False

    if all_passed:
        logger.info("All code quality checks passed!")
    else:
        logger.warning("Some code quality checks failed!")

    return all_passed


def format_code():
    """Format code using black and isort."""
    logger.info("Formatting code...")

    commands = [
        "black src/ main.py streamlit_app.py",
        "isort src/ main.py streamlit_app.py",
    ]

    for command in commands:
        if not run_command(command):
            return False

    logger.info("Code formatting completed!")
    return True


def build_docker():
    """Build Docker images."""
    logger.info("Building Docker images...")

    commands = [
        "docker build -t mews:latest --target production .",
        "docker build -t mews:dev --target development .",
    ]

    for command in commands:
        if not run_command(command):
            return False

    logger.info("Docker images built successfully!")
    return True


def deploy_local():
    """Deploy application locally using Docker Compose."""
    logger.info("Deploying locally...")

    commands = [
        "docker-compose up -d mews-app",
    ]

    for command in commands:
        if not run_command(command):
            return False

    logger.info("Local deployment completed!")
    logger.info("Application available at: http://localhost:8501")
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["data", "models", "outputs", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        # Create .gitkeep files to preserve empty directories
        gitkeep_file = Path(directory) / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()


def clean():
    """Clean up generated files and caches."""
    logger.info("Cleaning up...")

    patterns_to_remove = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        "*.log",
        "bandit-report.json",
        "safety-report.json",
    ]

    for pattern in patterns_to_remove:
        if os.name == "nt":  # Windows
            run_command(f"del /s /q {pattern}", cwd=".")
        else:  # Unix-like
            run_command(f"find . -name '{pattern}' -exec rm -rf {{}} +", cwd=".")

    logger.info("Cleanup completed!")


def main():
    """Main entry point for build script."""
    parser = argparse.ArgumentParser(description="MEWS Build and Deployment Script")
    parser.add_argument(
        "command",
        choices=[
            "setup",
            "setup-dev",
            "test",
            "lint",
            "format",
            "build",
            "deploy",
            "clean",
            "all",
        ],
        help="Command to execute",
    )

    args = parser.parse_args()

    # Create necessary directories
    create_directories()

    try:
        if args.command == "setup":
            success = setup_environment()
        elif args.command == "setup-dev":
            success = setup_dev_environment()
        elif args.command == "test":
            success = run_tests()
        elif args.command == "lint":
            success = run_linting()
        elif args.command == "format":
            success = format_code()
        elif args.command == "build":
            success = build_docker()
        elif args.command == "deploy":
            success = deploy_local()
        elif args.command == "clean":
            clean()
            success = True
        elif args.command == "all":
            success = (
                setup_dev_environment()
                and format_code()
                and run_linting()
                and run_tests()
                and build_docker()
            )
        else:
            parser.print_help()
            success = False

        if success:
            logger.info(f"'{args.command}' completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"'{args.command}' failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Build error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
