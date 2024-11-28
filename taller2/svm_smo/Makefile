# lazy makefile
PROG=procRecogNumber
all:
	mkdir -p bin
	matlab -nodisplay -nosplash -nodesktop -r "try run('SVMCode/HoGFunctions/make.m'); catch; end; quit"
	cp SVMCode/HoGFunctions/*.mex* bin/
	mcc -d bin -m ${PROG}.m -R -nodisplay -I SVMCode/ -I SVMCode/HoGFunctions/

