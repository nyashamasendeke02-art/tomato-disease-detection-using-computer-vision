# Structural Modifications

## 1. Added Directories
| Directory         | Purpose |
|-------------------|---------|
| `scripts/`        | Deployment scripts, data collection tools |
| `docs/`           | Project documentation, API references |
| `environments/`   | Environment configuration files (e.g., Docker, conda) |
| `pipeline/`       | CI/CD pipeline definitions (GitHub Actions, GitLab CI) |

## 2. Modified Files
### `README.md`
- Added a section linking to this `Modifications.md`
- Updated dependency installation instructions

## 3. New Files
- `environments/requirements.lock`: Frozen dependency versions
- `environments/Dockerfile`: Containerization configuration
- `pipeline/.github/workflows/training.yml`: Training pipeline automation

## 4. Structural Improvements
- Separated training scripts from evaluation scripts
- Added subdirectories for:
  - `src/data/collect/`: Data collection scripts
  - `src/models/deploy/`: Model serving code
