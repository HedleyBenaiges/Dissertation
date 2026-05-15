#!/bin/sh
# linker.ld = ELMO/Examples/ProjectTemplate/main.ld
# elmoasmfunctions.s = ELMO/Examples/ProjectTemplate/elmoasmfunctions.s

# 1. Compile all source files into object files
arm-none-eabi-gcc -Os -mthumb -mcpu=cortex-m0 -I. -I./common -c \
      elmo_test.c cbd.c kem.c indcpa.c ntt.c poly.c polyvec.c reduce.c \
      symmetric-shake.c verify.c ./common/fips202.c #kem.c indcpa.c (between cbd, ntt)

# 2. Link the objects with the assembly startup file
# Ignore the warnings about _close, _read, etc.
arm-none-eabi-gcc -T linker.ld -nostartfiles \
      --specs=nosys.specs --specs=nano.specs \
      elmoasmfunctions.s \
      elmo_test.o cbd.o indcpa.o ntt.o poly.o polyvec.o reduce.o \
      symmetric-shake.o verify.o fips202.o \
      -o elmo_target.elf # kem.o, indcpa.o

# 3. Strip to raw binary for ELMO
arm-none-eabi-objcopy -O binary elmo_target.elf elmo_target.bin
