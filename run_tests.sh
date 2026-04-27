#!/bin/bash
# Run TP unit tests (mp.spawn, no torchrun needed).
# Usage: bash run_tests.sh

echo "Running TP tests..."
pytest test_tp.py -s
