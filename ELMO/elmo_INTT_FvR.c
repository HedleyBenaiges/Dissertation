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
//  3. Uniform sampler (loop-free):
//     Two ELMO random bytes give a 16-bit value v; output v % q.
//     A rejection-sampling do...while loop caused ELMO to hang
//     permanently at the trace-1000 boundary (the emulator stalls
//     when randbyte is called inside a backward branch between the
//     fixed and random sets).  The mod approach is loop-free and
//     its ~5 % bias is negligible for TVLA purposes.
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
 * due to the mixing property of the NTT butterfly network.  Sampling
 * uniformly in [0, q-1] is therefore the correct model for the random
 * set of an invNTT leakage test.
 *
 * Implementation — loop-free mod reduction:
 *   Two ELMO random bytes give a 16-bit value v in [0, 65535].
 *   Output: v % KYBER_Q, which lies in [0, q-1].
 *
 *   Bias: 65536 = 19*3329 + 2415, so values [0, 2414] appear 20 times
 *   and values [2415, 3328] appear 19 times across the full 16-bit range.
 *   The relative bias is 1/19 ≈ 5.3 %, which is negligible for TVLA —
 *   the t-test only requires that the two sets differ in input; it does
 *   not require a perfectly flat distribution in the random set.
 *
 *   Crucially, this sampler contains NO branches that loop back, so
 *   ELMO's emulator cannot hang between the fixed and random sets.
 *   A rejection-sampling loop (do...while) caused ELMO to stall
 *   permanently at the trace-1000 boundary.
 */
static void sample_uniform_ntt_poly(poly *p) {
    int i;
    for (i = 0; i < KYBER_N; i++) {
        uint8_t lo, hi;
        uint16_t v;
        randbyte(&lo);
        randbyte(&hi);
        v = ((uint16_t)hi << 8) | (uint16_t)lo;
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
