// Fixed vs Random TVLA for ML-KEM-512 polyvec_basemul_acc_montgomery

#include <stdint.h>
#include <stddef.h>
#include "poly.h"     /* poly                                              */
#include "polyvec.h"  /* polyvec, PQCLEAN_MLKEM512_CLEAN_polyvec_basemul_acc_montgomery */
#include "params.h"   /* KYBER_N, KYBER_K, KYBER_Q                        */

// ELMO Triggers
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

// No. traces per set; total traces = NOTRACES * 2
#define NOTRACES 1000

/* Fixed NTT-domain coefficient value.
 *
 * 1665 = (KYBER_Q + 1) / 2 is chosen because:
 *   - It is the midpoint of the valid NTT coefficient range [0, q-1],
 *     meaning the Montgomery multiply path operates on operands whose
 *     Hamming weight is representative of the average case, rather than
 *     being biased toward extremes (0 collapses multiplications, q-1
 *     saturates them).
 *   - With both polyvec operands fixed to 1665, the internal computation
 *     montgomery_reduce((int32_t)1665 * 1665) = montgomery_reduce(2772225)
 *     exercises the full 32-bit multiply and modular reduction in the
 *     basemul butterfly without producing a degenerate zero or one result.
 *   - 1665 is coprime to KYBER_Q = 3329, so it has a well-defined
 *     multiplicative inverse and does not accidentally align with any
 *     root of unity in the NTT zeta table.                             */
#define FIXED_COEFF ((int16_t)1665)

/* ------------------------------------------------------------------ */
/* NTT-domain coefficient distribution for the random set             */
/* ------------------------------------------------------------------ */
/*
 * polyvec_basemul_acc_montgomery operates on two polyvecs whose
 * coefficients are in NTT domain after a prior poly_ntt() call.
 * After Barrett reduction, NTT coefficients lie in [0, q-1] = [0, 3328].
 *
 * Sampling strategy: take 12 bits from two randbyte() calls (only the
 * low nibble of the second byte is used) giving a value in [0, 4095].
 * A single conditional subtract brings values >= KYBER_Q into [0, q-1].
 * Values in [0, 766] appear at most twice (from [0,766] and [3329,4095]),
 * introducing a small bias that is negligible for TVLA purposes — the
 * t-test requires variation, not perfect uniformity.
 *
 * Both the a and b polyvecs are randomised independently so that the
 * product a[i] * b[i] inside each basemul butterfly varies across the
 * full joint input space, matching the real-world distribution of
 * (matrix A, secret s) pairs seen during ML-KEM-512 encapsulation.
 */

/* ------------------------------------------------------------------ */
/* Static allocation — avoids stack overflow on M0's small stack.     */
/*   2 x polyvec = 2 x KYBER_K x KYBER_N x sizeof(int16_t)           */
/*               = 2 x 2 x 256 x 2 = 2048 bytes                      */
/*   1 x poly    = 256 x 2          =  512 bytes                      */
/* ------------------------------------------------------------------ */
static polyvec a, b;
static poly    r;

static void set_fixed_polyvec(polyvec *pv) {
    int i, j;
    for (i = 0; i < KYBER_K; i++) {
        for (j = 0; j < KYBER_N; j++) {
            pv->vec[i].coeffs[j] = FIXED_COEFF;
        }
    }
}

static void sample_random_polyvec(polyvec *pv) {
    int i, j;
    uint8_t lo, hi;
    uint16_t v;
    for (i = 0; i < KYBER_K; i++) {
        for (j = 0; j < KYBER_N; j++) {
            randbyte(&lo);
            randbyte(&hi);
            /* 12-bit value: [0, 4095] → conditional subtract → [0, q-1] */
            v = (uint16_t)lo | ((uint16_t)(hi & 0x0Fu) << 8);
            if (v >= (uint16_t)KYBER_Q) {
                v -= (uint16_t)KYBER_Q;
            }
            pv->vec[i].coeffs[j] = (int16_t)v;
        }
    }
}

int main(void) {

    int i;

    /* FIXED input SET
     * Both polyvecs are filled with FIXED_COEFF every trace.
     * The Montgomery multiply and accumulate produce identical
     * intermediate values each time; power is constant across traces. */
    for (i = 0; i < NOTRACES; i++) {
        set_fixed_polyvec(&a);
        set_fixed_polyvec(&b);
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_polyvec_basemul_acc_montgomery(&r, &a, &b);
        endtrigger();
    }

    /* RANDOM input SET
     * Both polyvecs are independently re-sampled from [0, q-1] every
     * trace, varying both multiply operands in every basemul butterfly.
     * Data-dependent Hamming weight variation in the 32-bit products
     * and Montgomery remainders drives the detectable leakage signal. */
    for (i = 0; i < NOTRACES; i++) {
        sample_random_polyvec(&a);
        sample_random_polyvec(&b);
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_polyvec_basemul_acc_montgomery(&r, &a, &b);
        endtrigger();
    }

    /* Trigger Welch t-test across both sets of traces.
     * Results: output/fixedvsrandomtstatistics.txt
     * Cross-reference with: output/asmoutput/asmtrace00001.txt
     *
     * Flag instructions with |t| > 4.5 as leaking at the 99.999%
     * confidence level (FIXEDVSRANDOMFAIL threshold in elmodefines.h). */
    endprogram();
    return 0;
}
