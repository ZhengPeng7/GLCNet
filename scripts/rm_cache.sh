#!/bin/bash
# Remove cache files
rm -rf ckpts/

rm -rf __pycache__ */__pycache__ */*/__pycache__
rm -rf exp_*
rm -rf vis/
rm -rf data/
rm *.out

clear
