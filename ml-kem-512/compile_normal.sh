gcc -O3 -march=native \
        -I../../../common \
        *.c \
        ../../../common/fips202.c \
        ../../../common/randombytes.c \
        -o normal_test
