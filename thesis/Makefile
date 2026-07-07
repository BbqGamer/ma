FIGURES := $(wildcard figures/*.tex figures/*.png figures/*.pdf)
APPENDICES := $(wildcard appendices/*.tex)

.PHONY: all
all: build/main.pdf

build/main.pdf: bibliografia.bib main.tex ppfcmthesis.cls $(FIGURES) $(APPENDICES) | build
	pdflatex -output-directory build main.tex
	bibtex build/main
	pdflatex -output-directory build main.tex
	pdflatex -output-directory build main.tex

build:
	mkdir -p build
