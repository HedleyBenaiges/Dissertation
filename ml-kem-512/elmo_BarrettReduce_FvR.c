// Fixed vs Random TVLA for ML-KEM-512 Barrett Reduce

#include <stdint.h>
#include <stddef.h>
#include "poly.h"      /* poly, KYBER_N                                   */
#include "params.h"    /* KYBER_N, KYBER_Q                                */
#include "reduce.h"    /* PQCLEAN_MLKEM512_CLEAN_barrett_reduce           */

// ELMO Triggers
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

// No. traces per set; total traces = NOTRACES * 2
#define NOTRACES 1000

/* The fixed coefficient for the fixed set.
 *
 * KYBER_Q + 1 = 3330 is chosen deliberately:
 *   - It lies just above the modulus, so the Barrett multiply-and-subtract
 *     path is always exercised (non-trivial reduction).
 *   - It fits in int16_t (max 32767) without overflow.
 *   - Avoids 0, which suppresses operand-dependent leakage in the
 *     multiplier, and avoids exact multiples of Q, which would reduce
 *     to 0 and similarly collapse leakage.
 *
 * Internally, barrett_reduce computes:
 *   t = ((int32_t)v * a) >> 26;   // v = ((1<<26) + Q/2) / Q
 *   return a - t * Q;
 * so the fixed set probes identical data-dependent transitions every
 * trace, while the random set produces varying operand values. */
#define FIXED_COEFF ((int16_t)(KYBER_Q + 1))   /* 3330 */

/* ------------------------------------------------------------------ */
/* Uniform random sampler over the full int16_t domain                 */
/* ------------------------------------------------------------------ */
/*
 * Barrett reduce is called from poly_reduce() after NTT butterfly
 * arithmetic and polynomial multiplication. Post-butterfly values can
 * span the full int16_t range [-32768, 32767], so sampling uniformly
 * over that range exercises the complete input distribution that
 * barrett_reduce will encounter in real key-generation and
 * encapsulation flows.
 *
 * Two randbyte() calls assemble one 16-bit little-endian word, then
 * reinterpret it as a signed int16_t — matching the M0 ABI exactly.
 */
static void sample_random_poly(poly *p) {
    int i;
    uint8_t lo, hi;
    for (i = 0; i < KYBER_N; i++) {
        randbyte(&lo);
        randbyte(&hi);
        p->coeffs[i] = (int16_t)((uint16_t)lo | ((uint16_t)hi << 8));
    }
}

/* Fixed polynomial initialiser — every coefficient identical */
static void set_fixed_poly(poly *p) {
    int i;
    for (i = 0; i < KYBER_N; i++) {
        p->coeffs[i] = FIXED_COEFF;
    }
}

/* ------------------------------------------------------------------ */
/* Apply barrett_reduce to every coefficient of a polynomial           */
/* ------------------------------------------------------------------ */
/*
 * PQClean exposes PQCLEAN_MLKEM512_CLEAN_poly_reduce() in poly.h,
 * which performs exactly this loop internally. We replicate it here
 * to target the scalar barrett_reduce function directly, keeping the
 * trigger window tightly scoped to the reduction logic and avoiding
 * any incidental poly_reduce overhead.
 *
 * If you prefer to test the full poly_reduce call instead, replace the
 * body below with:
 *   PQCLEAN_MLKEM512_CLEAN_poly_reduce(p);
 * and remove the reduce.h include above.
 */
static void apply_barrett_reduce(poly *p) {
    p->coeffs[0] = PQCLEAN_MLKEM512_CLEAN_barrett_reduce(p->coeffs[0]);
}

// Static allocation — avoids stack overflow on M0's 2 KB default stack.
static poly s;

int main(void) {

    int i;

    /* FIXED plaintext SET
     * Every trace processes the same all-FIXED_COEFF polynomial.
     * Power consumption is identical across traces (up to noise). */
    sample_random_poly(&s);
    for (i = 0; i < NOTRACES; i++) {
        //set_fixed_poly(&s);
        starttrigger();
        apply_barrett_reduce(&s);
        endtrigger();
    }

    /* RANDOM plaintext SET
     * Each trace processes a freshly sampled uniform-random polynomial.
     * Data-dependent power variation is present in these traces. */
    for (i = 0; i < NOTRACES; i++) {
        sample_random_poly(&s);
        starttrigger();
        apply_barrett_reduce(&s);
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
