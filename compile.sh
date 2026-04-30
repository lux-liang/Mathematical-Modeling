#!/usr/bin/env bash
set -e

export TEXMFVAR="$PWD/.texmf-var"
export TEXMFCONFIG="$PWD/.texmf-config"
mkdir -p "$TEXMFVAR" "$TEXMFCONFIG"

xelatex main.tex
xelatex main.tex
