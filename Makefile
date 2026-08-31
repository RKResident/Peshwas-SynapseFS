# SynapseFS developer entry points.
#
# Every target clears PYTHONPATH. This is not cosmetic: a sourced ROS 2 setup
# exports PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages, which leaks
# Python 3.10 packages into this project's 3.14 venv and makes pytest crash
# while auto-loading ROS's `launch_testing` plugin. Clearing it per-command is
# more reliable than asking eight people to remember not to source ROS.

PY  := .venv/bin/python
RUN := PYTHONPATH= $(PY)

.PHONY: help test fixtures fixtures-small fixtures-all lint clean-fixtures network

help:
	@echo "make test            - run the test suite"
	@echo "make fixtures        - generate tiny mlp+cnn fixtures (fast, for unit tests)"
	@echo "make fixtures-small  - generate small mlp+cnn fixtures (for benchmarks)"
	@echo "make fixtures-all    - tiny + small, both architectures"
	@echo "make network        - build the C++ push/pull transfer tool"
	@echo "make clean-fixtures  - delete generated fixtures"

test:
	$(RUN) -m pytest

# Tiny fixtures are what the unit tests run against. Cheap enough to regenerate
# on demand, which is why they are gitignored rather than committed.
fixtures:
	$(RUN) tools/gen_fixtures.py --all --size tiny --out fixtures/

fixtures-small:
	$(RUN) tools/gen_fixtures.py --all --size small --out fixtures/

fixtures-all: fixtures fixtures-small

lint:
	$(RUN) -m compileall -q synapsefs tools tests

clean-fixtures:
	rm -rf fixtures/

# The transfer tool is C++ and standalone on purpose: a machine can host a repo
# without a Python environment. It is not part of the Python package -- there is
# no __init__.py, so setuptools does not pick it up.
network:
	g++ -std=c++17 -O2 -o synapsefs/networking/spp synapsefs/networking/spp.cpp
