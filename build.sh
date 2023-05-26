rm *.o;
g++ -std=c++14 -c -fPIC naive_blas.cpp;   
g++ -std=c++14 -c -fPIC utils.cpp;   
g++ -std=c++14 -o main.o main.cpp -L. naive_blas.o utils.o;
./main2.o;