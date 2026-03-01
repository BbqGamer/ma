build/main.pdf: bibliografia.bib main.tex figures/*.tex
	pdflatex -output-directory build main.tex
	bibtex build/main
	pdflatex -output-directory build main.tex
	pdflatex -output-directory build main.tex
