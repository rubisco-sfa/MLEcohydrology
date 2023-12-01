

photo: photo.f90
	gfortran photo.f90 -o photo

old: photo.f90
	gfortran photo.f90 -o photo \
	-I/home/nate/src/neural-fortran/build/include \
	-I/usr/lib64/gfortran/modules \
	-L/home/nate/src/neural-fortran/build/lib \
	-lneural -ljsonfortran -lfunctional -lh5fortran \
	-lhdf5 -lhdf5_fortran -lhdf5_hl -lhdf5hl_fortran

clean:
	rm -rf photo *~
