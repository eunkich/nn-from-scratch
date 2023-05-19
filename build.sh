g++ -std=c++14 -c -fPIC naive_blas.cpp;   
g++ -std=c++14 -o main.o main.cpp -L. naive_blas.o;
./main.o;
rm *.o;