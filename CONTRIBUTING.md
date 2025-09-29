# Contributing to Market Risk Early Warning System

Thank you for your interest in contributing to MEWS! This document provides guidelines for contributing to the project.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/mews-fin.git
   cd mews-fin
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   
   pip install -r requirements-dev.txt
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Download NLTK Data**
   ```bash
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
   ```

### Project Structure

```
mews-fin/
├── src/                 # Core modules
│   ├── config.py        # Configuration management
│   ├── data_fetcher.py  # Data collection
│   ├── ml_models.py     # Machine learning models
│   └── ...
├── tests/               # Test suite
├── docs/                # Documentation
├── .github/workflows/   # CI/CD pipelines
├── main.py              # CLI entry point
├── streamlit_app.py     # Web interface
└── requirements*.txt    # Dependencies
```

## 🔧 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow code style guidelines (Black, isort, flake8)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes
```bash
# Run tests
pytest tests/ -v

# Run linting
flake8 src/ main.py streamlit_app.py
black --check src/ main.py streamlit_app.py
isort --check-only src/ main.py streamlit_app.py

# Run type checking
mypy src/ --ignore-missing-imports

# Run security checks
bandit -r src/
safety check
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add new feature description"
```

Use conventional commit messages:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Create a Pull Request with:
- Clear description of changes
- Reference to related issues
- Screenshots for UI changes
- Performance impact notes

## 📝 Coding Standards

### Python Style Guide
- Follow PEP 8 with line length of 88 characters
- Use Black for code formatting
- Use isort for import sorting
- Add type hints where possible
- Write docstrings for all public functions

### Example Function:
```python
def calculate_risk_score(
    price_data: pd.DataFrame,
    sentiment_score: float,
    volatility_threshold: float = 0.02
) -> float:
    """
    Calculate risk score based on price data and sentiment.
    
    Args:
        price_data: Historical price data
        sentiment_score: News sentiment score (-1 to 1)
        volatility_threshold: Volatility threshold for risk calculation
        
    Returns:
        Risk score between 0 and 1
        
    Raises:
        ValueError: If price_data is empty
    """
    if price_data.empty:
        raise ValueError("Price data cannot be empty")
    
    # Implementation here
    return risk_score
```

### Testing Guidelines
- Write unit tests for all new functions
- Use pytest fixtures for test data
- Mock external APIs and dependencies
- Aim for >80% code coverage
- Include integration tests for complex workflows

### Example Test:
```python
def test_calculate_risk_score():
    """Test risk score calculation."""
    # Arrange
    price_data = create_sample_price_data()
    sentiment_score = 0.5
    
    # Act
    risk_score = calculate_risk_score(price_data, sentiment_score)
    
    # Assert
    assert 0 <= risk_score <= 1
    assert isinstance(risk_score, float)
```

## 🐛 Bug Reports

When reporting bugs, include:
- **Environment**: OS, Python version, package versions
- **Steps to Reproduce**: Detailed steps to trigger the bug
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full stack traces
- **Additional Context**: Screenshots, logs, etc.

### Bug Report Template:
```markdown
**Environment:**
- OS: Windows 10
- Python: 3.11.5
- Package Version: 1.0.0

**Steps to Reproduce:**
1. Run command `python main.py --symbols AAPL`
2. Wait for data fetching
3. Error occurs during preprocessing

**Expected Behavior:**
Should preprocess data without errors

**Actual Behavior:**
Raises KeyError for missing column

**Error Message:**
```
KeyError: 'Close'
```

**Additional Context:**
Happens only with certain symbols
```

## 💡 Feature Requests

For new features, provide:
- **Use Case**: Why is this feature needed?
- **Description**: What should the feature do?
- **Implementation Ideas**: How might it work?
- **Alternatives**: Other solutions considered?

## 📊 Performance Guidelines

- Profile code changes that might affect performance
- Avoid unnecessary data copies
- Use vectorized operations with pandas/numpy
- Cache expensive computations
- Consider memory usage for large datasets

## 🔒 Security Guidelines

- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all user inputs
- Follow security best practices for web interfaces
- Run security checks with bandit

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Include examples in documentation
- Update API documentation
- Add inline comments for complex logic

## 🧪 Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Cover edge cases and error conditions

### Integration Tests
- Test component interactions
- Use realistic test data
- Test end-to-end workflows

### Performance Tests
- Benchmark critical functions
- Test with large datasets
- Monitor memory usage

## 🚀 Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create pull request to main
6. Tag release after merge
7. GitHub Actions will handle deployment

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and ideas

## 🎯 Good First Issues

Look for issues labeled:
- `good first issue`: Perfect for newcomers
- `help wanted`: Community help needed
- `documentation`: Documentation improvements
- `testing`: Add or improve tests

Thank you for contributing to MEWS! 🎉
