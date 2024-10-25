# FeatureCircuitDiscovery
FeatureCircuitDiscovery is a library focused on exploring and experimenting with SAE feature circuits in transformer models.

## Getting Started
To set up your environment and start experimenting with FeatureCircuitDiscovery, follow these steps:

- Ensure you have [poetry](https://python-poetry.org/docs/#installation) installed on your system.
- Clone the FeatureCircuitDiscovery repository to your local machine.
- Navigate to the cloned repository directory and run `poetry install --with dev` to install all necessary dependencies.

``` bash
curl -sSL https://install.python-poetry.org | python3 - && export PATH="/root/.local/bin:$PATH"

git clone https://github.com/jl2191/feature-circuit-discovery.git

cd feature-circuit-discovery

poetry install --with dev
```

Poetry is configured to use system packages by default, which can be beneficial when working on systems with pre-installed packages like PyTorch. To change this behavior, set `options.system-site-packages` to `false` in `poetry.toml`.

## Contributing
Contributions are welcome! Here are some guidelines to follow:

- Type checking is enforced using [Pyright](https://github.com/microsoft/pyright). Please include type hints in all function definitions.
- Write tests using [Pytest](https://docs.pytest.org/en/stable/).
- Format your code with [Black](https://github.com/psf/black).
- Lint your code using [ruff](https://github.com/astral-sh/ruff).

To check / fix your code run:
```bash
pre-commit run --all-files
```
Install the git hook with:
``` bash
pre-commit install
```
To execute tests labeled as slow, run:
``` bash
pytest --runslow
```
To execute tests marked as slow and are benchmarked, run:
``` bash
pytest --runslow --benchmark-only
```
For these commands to run, you may need to run:
```bash
poetry shell
```
and/or
```bash
export PATH="/root/.local/bin:$PATH"
```
depending on your environment setup.

## Licensing

This repository is open-sourced under the MIT license. We warmly invite you to explore, modify, and distribute the software as you see fit. For more details, please refer to the LICENSE file in this repository.
