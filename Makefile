.PHONY: python rust test benchmark clean

# ----- Rust -----

rust:
	cargo build --workspace --release

# ----- Python -----

PY_PKG_DIR 	:= substrate-core-py
VENV_DIR 	:= .venv
PYTHON 		:= $(VENV_DIR)/bin/python
PIP 		:= $(VENV_DIR)/bin/pip
DOCS_DIR	:= docs/source
RUST_DOCS	:= target/doc
PYTHON_DOCS	:= target/doc/python

python-venv:
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install build

python-dev-deps: python-venv
	$(PIP) install -r requirements-dev.txt
	cd $(PY_PKG_DIR) && ../$(PYTHON) -m maturin develop

python-deps: python-venv
	$(PIP) install -r requirements.txt

python: python-deps python-dev-deps rust
	cd $(PY_PKG_DIR) && ../$(PYTHON) -m build

# ----- Test -----

rust-test:
	cargo test

python-test: python-dev-deps
	cd $(PY_PKG_DIR) && ../$(PYTHON) -m pytest tests

test: rust-test python-test

# ----- Benchmark -----

rust-benchmark:
	cargo bench -p substrate-core-impl

python-benchmark: python-dev-deps
	cd $(PY_PKG_DIR) && ../$(PYTHON) -m pytest benches

benchmark: rust-benchmark python-benchmark

# ----- Documentation -----

rust-documentation:
	cargo doc --workspace --no-deps

python-documentation: python
	$(PYTHON) -m sphinx -b html $(DOCS_DIR) $(PYTHON_DOCS)

documentation: rust-documentation python-documentation

documentation-open: documentation
	xdg-open docs/rust.html
	xdg-open docs/python.html

# ----- Format/Lint -----

rust-format:
	cargo fmt --all

python-format: python-dev-deps
	$(PYTHON) -m ruff format $(PY_PKG_DIR)

format: rust-format python-format

rust-lint:
	cargo clippy --workspace -- -D warnings

python-lint: python-dev-deps
	$(PYTHON) -m ruff check $(PY_PKG_DIR)

lint: rust-lint python-lint

# ----- Clean -----

clean:
	cargo clean
	rm -rf $(VENV_DIR)
	git clean -fdX