#!/bin/bash

patch_sizes=(3 6 12 16 32)

for patch in "${patch_sizes[@]}"; do
    python eq_modern_vit.py --modern_vit --patch_size $patch
done