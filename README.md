# Installation
```bash
git clone git@github.com:geoffwoollard/xgw.git
cd xgw
pip install -e .[dev] # installs using pyproject.toml
```

# Benchmarking
```bash
python -m pytest tests/test_main_algorithms.py --log-cli-level=INFO --log-cli-format="%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s" --log-cli-date-format="%Y-%m-%d %H:%M:%S"
```