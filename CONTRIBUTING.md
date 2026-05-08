# Contributing

## Setup

Clone the repo and install dependencies with [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/funkelab/motile_toolbox.git
cd motile_toolbox
uv sync
```

Install pre-commit hooks:

```
uv run pre-commit install
```

## Development

Run tests:

```
just test
```

Preview docs locally:

```
just watch-docs
```

Build docs to `docs/`:

```
just build-docs
```

## Before submitting a PR

Pre-commit hooks (ruff, mypy) will run automatically on commit if installed. You can also run them manually:

```
uv run pre-commit run --all-files
```
