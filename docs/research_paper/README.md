# Research Paper - LaTeX Source

## Files

- **`main.tex`**: Main paper source (IEEE conference format)
- **`references.bib`**: Bibliography with 50+ citations
- **`compile.sh`**: Compilation script
- **`00_PAPER_OUTLINE.md`**: Markdown outline (reference)
- **`BIBLIOGRAPHY.md`**: Citation database (reference)

## Quick Start

### Prerequisites

Install LaTeX distribution:
- **macOS**: `brew install --cask mactex`
- **Ubuntu**: `sudo apt-get install texlive-full`
- **Windows**: Install MiKTeX or TeX Live

### Compile the Paper

```bash
cd docs/research_paper/

# Option 1: Use the script
./compile.sh

# Option 2: Manual compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`

## Paper Structure

1. **Abstract** (250-300 words)
2. **Introduction**
   - Motivation
   - Contributions
   - Organization
3. **Related Work**
   - Humanoid robotics
   - NLP for robotics
   - Computer vision
   - Multimodal learning
   - Edge computing
   - MLOps
4. **System Architecture**
   - Design philosophy
   - Component architecture (7 services)
   - Cloud-edge distribution
5. **Implementation Details**
   - Technology stack
   - Model optimization
   - Hardware configuration
6. **Data Strategy and Training**
   - Dataset curation
   - Training methodology
   - Evaluation protocols
7. **MLOps Pipeline**
   - Data management
   - Continuous training
   - Monitoring
8. **Experimental Results**
   - Benchmarks
   - Perception performance
   - Language understanding
   - End-to-end tasks
   - Latency analysis
   - Safety validation
9. **Discussion**
   - Key findings
   - Limitations
   - Ethics
   - Future work
10. **Conclusion**

## Customization

### Add Figures

```latex
\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{figures/architecture.pdf}}
\caption{System architecture diagram.}
\label{fig:architecture}
\end{figure}
```

### Add Tables

```latex
\begin{table}[htbp]
\caption{Performance Comparison}
\label{tab:performance}
\centering
\begin{tabular}{lcc}
\toprule
Method & Accuracy & Latency (ms) \\
\midrule
Ours & 96.5\% & 42 \\
Baseline & 92.3\% & 67 \\
\bottomrule
\end{tabular}
\end{table}
```

### Add Citations

Reference exists in `references.bib`:
```latex
Recent work \cite{ahn2022saycan} demonstrated...
```

Add new citation to `references.bib`:
```bibtex
@article{author2024title,
  title={Paper Title},
  author={Author, First and Author, Second},
  journal={Conference/Journal Name},
  year={2024}
}
```

## TODO List

Before submission:

- [ ] Fill in abstract (currently placeholder)
- [ ] Add architecture diagram (Figure 1)
- [ ] Add performance tables (Section 7)
- [ ] Add ablation study results
- [ ] Review all TODO markers in main.tex
- [ ] Proofread for grammar and clarity
- [ ] Check citation formatting
- [ ] Verify all references are cited
- [ ] Add acknowledgments
- [ ] Final compilation and PDF check

## Target Venues

- **ICRA 2026**: Deadline ~September 2025
- **IROS 2026**: Deadline ~March 2026
- **CoRL 2025**: Deadline ~July 2025
- **NeurIPS 2025**: Deadline ~May 2025
- **RSS 2025**: Deadline ~January 2025

## Page Limit

IEEE conference format: **6-8 pages** (excluding references)

Current status: ~12 pages (needs condensing or splitting)

## Tips

1. **Figures**: Create high-quality vector graphics (PDF format preferred)
2. **Tables**: Keep simple and readable
3. **Citations**: Use `\cite{key}` consistently
4. **Math**: Use equation environment for important formulas
5. **Acronyms**: Define on first use (e.g., "Natural Language Processing (NLP)")

## Troubleshooting

**Missing packages?**
```bash
# macOS
sudo tlmgr update --self
sudo tlmgr install <package-name>

# Ubuntu
sudo apt-get install texlive-<package-collection>
```

**Bibliography not showing?**
Make sure to run: `pdflatex → bibtex → pdflatex → pdflatex`

**Citations showing as [?]?**
Check that the citation key in `\cite{}` matches `references.bib`

## Online Editors

If you prefer online editing:
- **Overleaf**: https://www.overleaf.com (upload `main.tex` and `references.bib`)
- **ShareLaTeX**: Similar to Overleaf

## Version Control

Track changes with Git:
```bash
git add main.tex references.bib
git commit -m "Updated introduction section"
```

## Collaboration

For multi-author collaboration:
1. Use Overleaf (real-time collaborative editing)
2. Or use Git branches for parallel work
3. Use `\todo{}` command to mark sections needing attention

---

**Author**: Victor Ibhafidon  
**Affiliation**: Xtainless Technologies  
**Last Updated**: October 2025
