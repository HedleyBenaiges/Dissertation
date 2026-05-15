# To run a test with ELMO
In ./ml-kem-512
    Copy a C script to elmo_test.c
    Run ./compile_elmo.sh
    This will create the file elmo_target.bin
For a Fixed vs Random test in ./ELMO run 
    Run ./elmo_fixedvsrandom_binary ../ml-kem-512/elmo_target.bin
    Or use an already existing .bin file
For plain power trace generation
    Run ./elmo ../ml-kem-512/elmo_target.bin
