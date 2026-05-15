// Fixed vs Random TVLA for ML-KEM-512 poly_invntt_tomont
//
// Structure mirrors elmo_NTT_FvR.c but targets the inverse NTT.
//
// Key differences from the NTT variant:
//
//  1. Function under test:
//       PQCLEAN_MLKEM512_CLEAN_poly_invntt_tomont(&s)
//     instead of poly_ntt. The invNTT converts a polynomial from
//     NTT domain back to the standard (time) domain, applying a
//     Montgomery reduction factor at the end (the "tomont" step).
//
//  2. Input domain:
//     The invNTT consumes NTT-domain coefficients, not CBD-sampled
//     secret coefficients.  In ML-KEM, NTT-domain coefficients are
//     (after Barrett/Montgomery reduction) approximately uniform in
//     [0, q-1].  The random set therefore samples from that
//     distribution rather than from CBD(eta1=3).
//
//  3. Uniform sampler:
//     Two ELMO random bytes give a 16-bit value in [0, 65535].
//     Rejection of values >= 19*q (= 63251) yields a perfectly
//     unbiased draw from [0, q-1]; the rejection rate is ~3.5 %
//     (< 1.04 randbyte pairs expected per coefficient on average).
//     Rejection sampling ensures the random and fixed distributions
//     differ only by the secret-dependent value, keeping the Welch
//     t-test statistically clean.
//
// ELMO setup (elmodefines.h):
//   - FIXEDVSRANDOM must be defined
//   - BINARYTRACES recommended (speed + storage)
//   - DIFFTRACELENGTH must NOT be defined (all traces are the same
//     length; the invNTT path is data-independent for a fixed-size
//     polynomial)
//
// Build and run:
//   arm-none-eabi-gcc -mcpu=cortex-m0 -mthumb -O2 \
//       -I path/to/pqclean/crypto_kem/ml-kem-512/clean \
//       -o elmo_INTT_FvR.elf elmo_INTT_FvR.c \
//       path/to/pqclean/.../poly.c ...
//   arm-none-eabi-objcopy -O binary elmo_INTT_FvR.elf elmo_INTT_FvR.bin
//   ./elmo elmo_INTT_FvR.bin
//
// Results:
//   output/fixedvsrandomtstatistics.txt  — t-statistic per instruction
//   output/asmoutput/asmtrace00001.txt   — instruction trace for
//                                          cross-referencing leaky points

#include <stdint.h>
#include <stddef.h>
#include "poly.h"      /* poly, PQCLEAN_MLKEM512_CLEAN_poly_invntt_tomont */
#include "params.h"    /* KYBER_N, KYBER_Q                                */

/* ------------------------------------------------------------------ */
/* ELMO triggers and helpers                                           */
/* ------------------------------------------------------------------ */
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

/* Number of traces in each half (fixed / random).
 * Total traces passed to ELMO = 2 * NOTRACES = 2000.
 * Increase to 5000–10000 for a more powerful test (lower detection
 * threshold), at the cost of longer simulation time.               */
#define NOTRACES 1000

/* Fixed NTT-domain coefficient used for every trace in the fixed set.
 *
 * Chosen as 1 for the same reasons as the NTT variant:
 *   - Non-zero: avoids trivially zeroing Montgomery multiplications
 *     inside the butterfly units, which would suppress observable
 *     power variation.
 *   - Small and realistic: NTT-domain values of 1 (and its negative,
 *     q-1 = 3328) naturally arise from constant or near-constant
 *     input polynomials.
 *   - Not a special root or twiddle factor value that might cause
 *     arithmetic cancellation.
 *
 * Alternative: KYBER_Q/2 = 1664 exercises mid-range arithmetic and
 * can be substituted if a different operating point is desired.      */
#define FIXED_COEFF 1

/* ------------------------------------------------------------------ */
/* Uniform sampler for NTT-domain coefficients in [0, q-1]           */
/* ------------------------------------------------------------------ */
/*
 * The input to poly_invntt_tomont is a polynomial in NTT representation
 * whose coefficients lie in [0, q-1] (stored as positive representatives
 * in int16_t, consistent with PQClean's convention).
 *
 * In a real ML-KEM encapsulation or decapsulation, these coefficients
 * are the NTT of a CBD-sampled secret or error polynomial.  Their
 * marginal distribution after the NTT is close to uniform over [0, q-1]
 * due to the mixing property of the NWT butterfly network.  Sampling
 * uniformly in [0, q-1] is therefore the correct model for the random
 * set of an invNTT leakage test.
 *
 * Rejection sampling:
 *   Two ELMO random bytes → 16-bit value v in [0, 65535].
 *   Accept iff v < 19 * KYBER_Q (= 63251); output v mod KYBER_Q.
 *   Rejection rate: (65536 - 63251) / 65536 ≈ 3.49 %.
 *   Expected randbyte pairs per coefficient: 1 / (1 - 0.0349) ≈ 1.036.
 *   This is perfectly unbiased — every residue in [0, q-1] is equally
 *   likely — unlike the biased 12-bit-and-subtract approach.
 */
static void sample_uniform_ntt_poly(poly *p) {
    int i;
    for (i = 0; i < KYBER_N; i++) {
        uint8_t lo, hi;
        uint16_t v;
        /* Rejection loop — terminates in O(1) iterations on average. */
        do {
            randbyte(&lo);
            randbyte(&hi);
            v = ((uint16_t)hi << 8) | (uint16_t)lo;
        } while (v >= (uint16_t)(19 * KYBER_Q)); /* 19*3329 = 63251 */
        p->coeffs[i] = (int16_t)(v % (uint16_t)KYBER_Q);
    }
}

/* ------------------------------------------------------------------ */
/* Fixed polynomial initialiser                                        */
/* ------------------------------------------------------------------ */
static void set_fixed_poly(poly *p) {
    int i;
    for (i = 0; i < KYBER_N; i++) {
        p->coeffs[i] = (int16_t)FIXED_COEFF;
    }
}

/* Static allocation — avoids stack overflow on M0's small stack.    */
static poly s;

/* ------------------------------------------------------------------ */
/* Main: fixed set first, then random set (required by ELMO FvR)     */
/* ------------------------------------------------------------------ */
int main(void) {

    int i;

    /* ---- FIXED plaintext SET ---- */
    /* Every polynomial is identical: all coefficients == FIXED_COEFF.
     * ELMO groups these as the first NOTRACES triggers for the fixed
     * half of the t-test.                                            */
    for (i = 0; i < NOTRACES; i++) {
        set_fixed_poly(&s);
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_poly_invntt_tomont(&s);
        endtrigger();
    }

    /* ---- RANDOM plaintext SET ---- */
    /* Each polynomial has fresh coefficients drawn uniformly from
     * [0, q-1], matching the true distribution of NTT-domain inputs.
     * ELMO groups these as the second NOTRACES triggers for the random
     * half of the t-test.                                            */
    for (i = 0; i < NOTRACES; i++) {
        sample_uniform_ntt_poly(&s);
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_poly_invntt_tomont(&s);
        endtrigger();
    }

    /* Signal ELMO to run the Welch t-test across both sets of traces.
     * Results: output/fixedvsrandomtstatistics.txt
     * Any instruction with |t| > 4.5 is flagged as leaking at the
     * 99.999 % confidence level (for large N).
     * Cross-reference with: output/asmoutput/asmtrace00001.txt      */
    endprogram();
    return 0;
}
