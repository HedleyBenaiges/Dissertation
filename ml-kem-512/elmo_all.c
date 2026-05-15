#include <stdint.h>
#include <stddef.h>
#include "api.h"
#include "poly.h" // for Poly_NTT
#include "params.h" // for KYBER_N

#define starttrigger() *((volatile uint32_t *)0xE0000004) = 1
#define endtrigger()   *((volatile uint32_t *)0xE0000004) = 0
#define endprogram()   *((volatile uint32_t *)0xF0000000) = 0

//#define readbyte(b)   (*((volatile uint8_t *)0xE0000000) = (b))
// Change 0xE1000000 to 0xE1000004 to trigger the file write in elmo.c
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))
#define printbyte(addr)  (*((volatile uint8_t *)0xE0000000) = *(addr))

// Ensure the name matches what PQClean expects internally
void PQCLEAN_randombytes(uint8_t *out, size_t outlen) {
    for (size_t i = 0; i < outlen; i++) {
        randbyte(&out[i]);
    }
}

int main() {
    starttrigger();
    unsigned char pk[PQCLEAN_MLKEM512_CLEAN_CRYPTO_PUBLICKEYBYTES];
    endtrigger();
    starttrigger();
    unsigned char sk[PQCLEAN_MLKEM512_CLEAN_CRYPTO_SECRETKEYBYTES];
    endtrigger();
    endprogram();
}
