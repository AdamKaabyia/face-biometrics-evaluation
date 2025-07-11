# Face Biometrics Evaluation System

A comprehensive face recognition evaluation system that compares multiple face recognition methods and evaluates their vulnerability to morphing attacks.

## Quick Start (Automatic Setup)

The easiest way to use this system is with the automated `run.py` script:

```bash
# 1. Clone the repository
git clone https://github.com/AdamKaabyia/face-biometrics-evaluation.git
cd face-biometrics-evaluation

# 2. Prepare your dataset
# Place face images in Data/Face/ organized by person: Data/Face/person_001/, Data/Face/person_002/, etc.

# 3. One-time setup (installs everything automatically)
python run.py setup

# 4. Run complete evaluation
python run.py all
```

That's it! The system will automatically create the environment, install dependencies, and run all evaluations.

## All Available Commands

```bash
# Setup and Environment
python run.py setup               # First-time setup (creates venv, installs dependencies)
python run.py status              # Check system and project status
python run.py clean               # Clean temporary files and cache
python run.py reset               # Remove virtual environment (for fresh start)

# Data Management
python run.py check-data          # Validate dataset organization
python run.py results             # Show existing results

# Face Recognition Evaluation
python run.py roc                 # Generate static ROC curves (PNG)
python run.py html                # Generate interactive HTML ROC curves
python run.py morph               # Run morphing attack evaluation
python run.py all                 # Run complete evaluation (roc + html + morph)

# Interactive Mode
python run.py interactive         # Start interactive menu
python run.py help                # Show detailed help and examples
```

### Interactive Mode

For a guided experience, use interactive mode:

```bash
python run.py interactive
```

This will show you a menu:
```
Available Commands:
  1) Setup environment
  2) Run face recognition evaluation
  3) Generate interactive HTML ROC curves
  4) Run morphing attack evaluation
  5) Run complete evaluation
  6) Show results
  7) Check status
  8) Clean temporary files
  9) Reset project
  q) Quit interactive mode

Enter command number (or 'q' to quit):
```

## Requirements

- Python 3.11+ (recommended) or Python 3.8+
- Windows 10/11, Linux, or macOS
- At least 4GB RAM
- 2GB free disk space

## Dataset Setup

Organize your face images in this structure:

```
Data/Face/
├── person_001/
│   ├── image1.bmp
│   ├── image2.bmp
│   └── ...
├── person_002/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

**Supported formats**: BMP, JPG, JPEG, PNG
**Minimum**: 2 images per person for pair generation

## Manual Setup (Advanced Users)

If you prefer manual control over the environment:

### Windows
```cmd
# Create virtual environment
python -m venv face-biometrics-venv

# Activate environment
face-biometrics-venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run evaluations directly
python roc.py
python generate_html_roc.py
python morph.py
```

### Linux/macOS
```bash
# Create virtual environment
python -m venv face-biometrics-venv

# Activate environment
source face-biometrics-venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run evaluations directly
python roc.py
python generate_html_roc.py
python morph.py
```

## Overview

This project implements a face biometrics evaluation system that:

1. **Evaluates multiple face recognition methods**:
   - PyTorch FaceNet (InceptionResnetV1)
   - DeepFace FaceNet
   - OpenCV grayscale baseline

2. **Analyzes performance metrics**:
   - ROC curves and AUC scores
   - False Match Rate (FMR) and False Non-Match Rate (FNMR)
   - Threshold analysis at different FMR targets

3. **Tests morphing attack vulnerability**:
   - Generates face morphs using Delaunay triangulation
   - Evaluates attack success rates across different matchers
   - Provides detailed attack analysis and statistics

## Project Structure

```
face-biometrics-evaluation/
├── run.py               # Cross-platform automated runner
├── roc.py               # Static ROC curve evaluation
├── generate_html_roc.py # Interactive HTML ROC curve generator
├── morph.py             # Morphing attack evaluation
├── logger.py            # Centralized logging configuration
├── utils.py             # Shared utility functions and constants
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── Data/               # Dataset directory
│   └── Face/           # Face images organized by person ID
│       ├── person_001/ # Individual person directories
│       ├── person_002/
│       └── ...
└── results/            # Output directory for results
    ├── roc_comparison.png      # Static ROC curves
    ├── index.html              # Interactive HTML ROC curves
    ├── fmr_fnmr_table.csv      # Performance metrics
    ├── morph_attack_pairs.csv  # Per-pair attack results
    └── morph_attack_summary.json # Attack statistics
```

## Output Files

### Face Recognition Evaluation
- **`results/roc_comparison.png`**: Static ROC curve comparison
- **`results/index.html`**: Interactive HTML ROC curves with hover details and zoom
- **`results/fmr_fnmr_table.csv`**: Detailed performance metrics table

### Morphing Attack Evaluation
- **`results/morph_attack_pairs.csv`**: Per-pair attack results with detailed logs
- **`results/morph_attack_summary.json`**: Overall attack success statistics

## Configuration

### Main Configuration (`utils.py`)

```python
# Dataset and output directories
DATASET_DIR = os.path.join('Data', 'Face')
OUTPUT_DIR = 'results'

# Evaluation parameters
FMR_TARGETS = [0.001, 0.01, 0.05]  # Target false match rates
LOG_EVERY = 100                     # Progress logging frequency
NUM_THREADS = 4                     # Parallel processing threads
```

### Morphing Attack Configuration (`morph.py`)

```python
# Morphing parameters
CANVAS_SIZE = (512, 512)    # Image processing size
MAX_PAIRS = 100             # Maximum pairs to evaluate
ALPHA = 0.5                 # Morphing blend factor

# Attack thresholds
THRESHOLDS = {
    "pytorch": 0.70,        # PyTorch FaceNet threshold
    "deepface": 0.70,       # DeepFace threshold
    "opencv_gray": -0.50    # OpenCV grayscale threshold
}
```

## Results Interpretation

### Face Recognition Results

The system generates:

1. **ROC Curves**: Visual comparison of different face recognition methods
2. **Performance Table**: Detailed metrics including:
   - Tool name
   - Target FMR
   - Operating threshold
   - Actual FMR achieved
   - Corresponding FNMR

### Morphing Attack Results

The morphing attack evaluation provides:

1. **Attack Success Rates**: Percentage of successful attacks per method
2. **Detailed Logs**: Per-pair attack outcomes
3. **Statistical Summary**: Overall attack effectiveness

## Development

### Code Structure

- **`run.py`**: Cross-platform CLI interface with automated environment management
- **`logger.py`**: Centralized logging with configurable levels
- **`utils.py`**: Shared utilities including dataset loading, model initialization, embedding extraction
- **`roc.py`**: Main evaluation pipeline (static PNG output)
- **`generate_html_roc.py`**: Interactive HTML ROC curve generator using Plotly
- **`morph.py`**: Morphing attack implementation with Delaunay triangulation

### Adding New Methods

To add a new face recognition method:

1. **Implement embedding function** in `utils.py`:
```python
def embed_new_method(path: str, model) -> np.ndarray:
    # Your implementation here
    pass
```

2. **Add to main evaluation** in `roc.py`:
```python
# Add to caching section
new_cache = cache_new_method_embeddings(paths, model)

# Add to scoring section
scores['NewMethod'] = []
for i, (p1, p2) in enumerate(pairs, 1):
    e1, e2 = new_cache[p1], new_cache[p2]
    scores['NewMethod'].append(cosine_similarity(e1, e2))
```

## Troubleshooting

### Common Issues

#### All Platforms
1. **Python version compatibility**: Ensure Python 3.8+ is installed
2. **Memory issues**: Reduce `NUM_THREADS` or `MAX_PAIRS` in configuration
3. **Missing dataset**: Ensure face images are properly organized in `Data/Face/` directory

#### Windows-Specific
1. **PowerShell execution policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
2. **Long path names**: Enable long path support in Windows settings
3. **Antivirus interference**: Exclude project folder from real-time scanning

#### Linux/macOS-Specific
1. **Permission denied**: Use `chmod +x run.py` to make script executable
2. **Python 3.11 not found**: Install specific Python version using your package manager

### Performance Optimization

- **Parallel processing**: Adjust `NUM_THREADS` based on CPU cores
- **Memory usage**: Monitor RAM usage during embedding caching
- **Dataset size**: Use subset for testing by modifying `MAX_PAIRS`

### Getting Help

Use the system status command to diagnose issues:
```bash
python run.py status
```

This will show:
- Operating system information
- Python version and path
- Available disk space
- Virtual environment status
- Dataset status
