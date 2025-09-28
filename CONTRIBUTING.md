# Contributing to Alzheimer's Disease Detection Project

We welcome contributions to improve this project! Please read this guide before contributing.

## How to Contribute

### 1. Fork the Repository
- Fork the repository on GitHub
- Clone your fork locally
- Set up the upstream remote

```bash
git clone https://github.com/yourusername/Yasin-EOAD-LLM-train.git
cd Yasin-EOAD-LLM-train
git remote add upstream https://github.com/originalowner/Yasin-EOAD-LLM-train.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
```

### 4. Make Your Changes

- Write clean, well-documented code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 5. Test Your Changes

```bash
# Run tests
pytest

# Run linting
flake8 .

# Run type checking
mypy .
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of your changes"
```

Use conventional commit messages:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for updates to existing features
- `Remove:` for removing features
- `Docs:` for documentation changes
- `Test:` for test changes

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused
- Use meaningful variable names

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions
- Include examples in docstrings
- Update type hints

### Testing
- Write unit tests for new functionality
- Aim for good test coverage
- Test edge cases
- Use descriptive test names

## Project Structure

```
Yasin-EOAD-LLM-train/
├── main.py                 # Main pipeline script
├── config.py              # Configuration settings
├── data_preprocessing.py   # Data loading and preprocessing
├── train_model.py         # Model training
├── evaluate_model.py      # Model evaluation
├── inference.py           # Real-time inference
├── explainability.py      # Model explanations
├── tests/                 # Test files
├── docs/                  # Documentation
├── examples/              # Example scripts
└── notebooks/             # Jupyter notebooks
```

## Areas for Contribution

### High Priority
- [ ] Add more evaluation metrics
- [ ] Improve model calibration
- [ ] Add support for more languages
- [ ] Optimize inference speed
- [ ] Add more explainability methods

### Medium Priority
- [ ] Add data augmentation techniques
- [ ] Implement ensemble methods
- [ ] Add web interface
- [ ] Improve documentation
- [ ] Add more visualization tools

### Low Priority
- [ ] Add Docker support
- [ ] Create API endpoints
- [ ] Add model versioning
- [ ] Create tutorials

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: OS, Python version, package versions
6. **Screenshots**: If applicable
7. **Logs**: Relevant error messages or logs

## Feature Requests

When requesting features, please include:

1. **Use Case**: Why this feature would be useful
2. **Proposed Solution**: How you think it should work
3. **Alternatives**: Other solutions you've considered
4. **Additional Context**: Any other relevant information

## Code Review Process

1. All submissions require review
2. At least one maintainer must approve
3. All CI checks must pass
4. Code must follow style guidelines
5. Tests must be included for new features

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Open a new issue with the "question" label
3. Contact the maintainers

Thank you for contributing to this project!
