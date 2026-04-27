#!/bin/bash
# Run profile: no TP, TP, TP+overlap (all via mp.spawn, no torchrun needed).
# Usage: bash run_profile.sh

echo "Running profile..."
python profile_memory.py
