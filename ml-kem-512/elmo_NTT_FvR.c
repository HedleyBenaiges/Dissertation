// Fixed vs Random TVLA for ML-KEM-512 NTT(s)

#include <stdint.h>
#include <stddef.h>
#include "poly.h"      /* poly, PQCLEAN_MLKEM512_CLEAN_poly_ntt */
#include "params.h"    /* KYBER_N, KYBER_ETA1, KYBER_Q          */

// ELMO Triggers
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

// No. traces * 2 = total traces
#define NOTRACES 1000

/* The fixed coefficient value used for every trace in the fixed set.
 * Chosen as 1: a realistic CBD output (most common nonzero value for
 * eta=3), non-trivial enough to exercise butterfly arithmetic.
 * Avoid 0: it collapses many NTT multiplications to zero and
 * artificially suppresses observable leakage. */
#define FIXED_COEFF 1

/* ------------------------------------------------------------------ */
/* CBD(eta=3) sampler — matches the real distribution of s in KeyGen  */
/* ------------------------------------------------------------------ */
/*
 * For ML-KEM-512, secret polynomial s has coefficients from CBD(eta1=3):
 * sum 3 pairs of random bits, giving values in [-3, +3].
 * Using this distribution for the random set means the t-test detects
 * leakage under the same conditions as a real key recovery attack.
 *
 * Stored as positive representatives in [0, q-1] by adding KYBER_Q to
 * negative values — consistent with PQClean's poly_cbd_eta1() convention.
 */

static void sample_cbd_poly(poly *p) {
    int i, k;
    for (i = 0; i < KYBER_N; i++) {
        uint8_t a, b;
        int16_t coeff = 0;
        randbyte(&a);
        randbyte(&b);
        // coeff = popcount(a[0..eta-1]) - popcount(b[0..eta-1])
        for (k = 0; k < KYBER_ETA1; k++) {
            coeff += (int16_t)((a >> k) & 1);
            coeff -= (int16_t)((b >> k) & 1);
        }
        // Map [-3, +3] to [0, q-1]
        if (coeff < 0) {
            coeff += (int16_t)KYBER_Q;
        }
        p->coeffs[i] = coeff;
    }
}

// Fixed polynomial initialiser
static void set_fixed_poly(poly *p) {
    int i;
    for (i = 0; i < KYBER_N; i++) {
        p->coeffs[i] = (int16_t)FIXED_COEFF;
    }
}

// Static allocation — avoids stack overflow on M0's small stack.
static poly s;

int main(void) {
    int i;

    // FIXED plaintext SET
    for (i = 0; i < NOTRACES; i++) {
        set_fixed_poly(&s);
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_poly_ntt(&s);
        endtrigger();
    }

    //RANDOM plaintext SET
    for (i = 0; i < NOTRACES; i++) {
        sample_cbd_poly(&s);
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_poly_ntt(&s);
        endtrigger();
    }

    endprogram();
    return 0;
}
