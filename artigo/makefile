SRC=main
OBJ=*.aux *.bbl *.idx *.log *.lot *.pdf *.toc *.bak

RM=rm -f

build:
	pdflatex $(SRC) && bibtex $(SRC) && pdflatex $(SRC) && pdflatex $(SRC)

clean:
	$(RM) $(OBJ)
