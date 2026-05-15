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

static uint8_t pk[PQCLEAN_MLKEM512_CLEAN_CRYPTO_PUBLICKEYBYTES];
static uint8_t sk[PQCLEAN_MLKEM512_CLEAN_CRYPTO_SECRETKEYBYTES];

static poly my_polynomial;

int main(void) {
    // Test if randomness works
    uint8_t test_buffer[10];
    PQCLEAN_randombytes(test_buffer, 10);

    // Target: Just the NTT
    // 10 iterations to get 10 separate traces for SCA averaging
    for(int i = 0; i < 10; i++) {
        for(int j=0; j<KYBER_N; j++) {
            uint8_t random_val;
            randbyte(&random_val);
            my_polynomial.coeffs[j] = (int16_t)random_val;
        }


        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_poly_ntt(&my_polynomial);
        endtrigger(); 
    }

    endprogram(); 
    return 0;
}
