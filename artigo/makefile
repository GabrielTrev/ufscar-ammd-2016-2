SRC=main
OBJ=*.aux *.idx *.log *.lot *.pdf *.toc *.bak

RM=rm -f

build:
	pdflatex $(SRC) && bibtex $(SRC) && pdflatex $(SRC) && pdflatex $(SRC)

clean:
	$(RM) $(OBJ)
