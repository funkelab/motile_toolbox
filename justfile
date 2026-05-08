test:
    uv run pytest -v --cov=motile_toolbox --cov-report=term-missing

build-docs:
    uv run pdoc -o docs/ -d google src/motile_toolbox

watch-docs:
    uv run pdoc --docformat google src/motile_toolbox
