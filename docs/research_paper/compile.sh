#!/bin/bash
# ==============================================================================
# LaTeX Compilation Script for Research Paper
# ==============================================================================

echo "Compiling research paper..."

# Clean previous builds
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.pdf

# First pass
pdflatex main.tex

# BibTeX for references
bibtex main

# Second pass (resolve references)
pdflatex main.tex

# Third pass (resolve cross-references)
pdflatex main.tex

# Clean auxiliary files (optional)
# Uncomment to keep only the PDF
# rm -f *.aux *.log *.bbl *.blg *.out *.toc

echo "✅ Compilation complete! Output: main.pdf"

# Open the PDF (macOS)
open main.pdf 2>/dev/null || echo "PDF created: main.pdf"
