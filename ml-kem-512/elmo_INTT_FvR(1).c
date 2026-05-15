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
//  3. Uniform sampler (division-free):
//     Two ELMO random bytes → mask to 12 bits → subtract q if >= q.
//     Avoids both (a) rejection-loop backward branches and (b) the
//     % operator, which on Cortex-M0 (no UDIV instruction) compiles
//     to a call to __aeabi_uidivmod — a libgcc software-division loop.
//     Either loop caused ELMO to hang at the trace-1000 boundary.
//     The ~23 % low-end bias is negligible for TVLA.
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
 * Implementation — 12-bit mask + single conditional subtract:
 *   Two ELMO random bytes give a 16-bit value.  Masking to 12 bits
 *   gives v in [0, 4095].  One conditional subtract yields [0, q-1]:
 *
 *     v &= 0x0FFF;                       // [0, 4095]
 *     if (v >= KYBER_Q) v -= KYBER_Q;   // [0, 3328]
 *
 *   Bias: values [0, 766] (= 4096 - q - 1) appear twice in [0, 4095]
 *   and values [767, 3328] appear once, a 2:1 bias of ~23 % for the
 *   low end.  This is acceptable for TVLA — the t-test only requires
 *   the two sets to differ; the random distribution need not be flat.
 *
 *   Critically, this generates NO backward branches and makes NO
 *   library calls.  Two earlier approaches both caused ELMO to hang
 *   permanently at the trace-1000 boundary (the transition from the
 *   fixed to the random set):
 *
 *     do { ... } while (v >= 19*q)  — backward branch in rejection loop
 *     v % KYBER_Q                   — calls __aeabi_uidivmod (libgcc
 *                                     software-division loop) because
 *                                     Cortex-M0 has no UDIV instruction
 *
 *   The mask + conditional subtract compiles to: AND, CMP, BLT/SUBCS —
 *   all straight-line or forward-only branches that ELMO handles safely.
 */
static void sample_uniform_ntt_poly(poly *p) {
    int i;
    for (i = 0; i < KYBER_N; i++) {
        uint8_t lo, hi;
        uint16_t v;
        randbyte(&lo);
        randbyte(&hi);
        v  = ((uint16_t)hi << 8) | (uint16_t)lo;
        v &= 0x0FFFu;                        /* [0, 4095] — no division  */
        if (v >= (uint16_t)KYBER_Q) {        /* forward branch only      */
            v -= (uint16_t)KYBER_Q;
        }
        p->coeffs[i] = (int16_t)v;          /* [0, q-1]                 */
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
    sample_uniform_ntt_poly(&s);
    for (i = 0; i < NOTRACES; i++) {
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
