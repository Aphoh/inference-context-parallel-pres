ARTICLE  = context_parallel_inference
# ARTICLE  = paper
all:
	pdflatex ${ARTICLE}
	bibtex ${ARTICLE}
	pdflatex ${ARTICLE}
	pdflatex ${ARTICLE}
clean:
	rm -f context_parallel_inference.pdf *.aux *.log *.snm *.nav *.fls *.bbl *.blg *.fdb_latexmk *.out *.toc *.brf *.bcf *.run.xml *.idx *.ilg *.ind *.bak *~
