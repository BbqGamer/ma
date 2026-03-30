build/main.pdf: bibliografia.bib main.tex ppfcmthesis.cls figures/*.tex appendices/*.tex
	pdflatex -output-directory build main.tex
	bibtex build/main
	pdflatex -output-directory build main.tex
	pdflatex -output-directory build main.tex
