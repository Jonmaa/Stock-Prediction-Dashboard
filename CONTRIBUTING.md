# Contributing to Stock Prediction Dashboard

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear title** describing the enhancement
- **Detailed description** of the proposed feature
- **Use case** explaining why this would be useful
- **Possible implementation** if you have ideas

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Stock-Prediction-Dashboard.git
cd Stock-Prediction-Dashboard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov flake8 black

# Run tests
pytest tests/ -v

# Run linting
flake8 app/ --max-line-length=120

# Format code
black app/
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Write unit tests for new features

## Project Structure Guidelines

- **app/**: Streamlit application code
- **notebooks/**: Jupyter notebooks for analysis
- **data/**: Data files (don't commit large files)
- **models/**: Trained model files
- **tests/**: Unit tests

## Questions?

Feel free to open an issue with the "question" label.

Thank you for contributing! 🙏
