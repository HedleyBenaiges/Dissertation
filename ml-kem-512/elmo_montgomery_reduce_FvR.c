// Fixed vs Random TVLA for ML-KEM-512 montgomery_reduce
//
// Tests montgomery_reduce() in isolation — the Montgomery modular
// reduction at the heart of every NTT and invNTT butterfly multiply.
//
// montgomery_reduce(a) takes int32_t a and returns int16_t r such that:
//   r ≡ a * R^{-1}  (mod q),   |r| ≤ q/2
// where R = 2^16, q = KYBER_Q = 3329, QINV = q^{-1} mod 2^16.
//
// Why test this function directly:
//   The NTT TVLA showed max |t| = 87.8 across the whole NTT, but that
//   conflates every instruction in every butterfly.  Testing
//   montgomery_reduce in isolation identifies whether the reduction
//   step itself leaks — i.e. whether the intermediate value
//   u = (int16_t)(a * QINV) or the final shift (a - u*q) >> 16
//   carries data-dependent power variation.
//
// Trace structure:
//   KYBER_N (256) back-to-back calls per trigger, matching the number
//   of montgomery_reduce invocations per NTT layer.  This gives a
//   trace long enough for reliable TVLA statistics while keeping
//   simulation time reasonable.
//
// Random input generation:
//   3 randbyte calls → 24-bit unsigned value in [0, 16,777,215].
//   This always satisfies montgomery_reduce's contract (|a| ≤ q*2^15
//   = 109,166,592).  No division, no modulo, no rejection loop —
//   avoiding the two failure modes that caused ELMO to hang at the
//   fixed/random boundary in previous attempts:
//     (a) do...while with randbyte → backward branch stall
//     (b) v % KYBER_Q            → __aeabi_uidivmod software loop
//   All randbyte calls happen OUTSIDE the trigger so they do not
//   appear in the measured trace window.
//
// ELMO setup (elmodefines.h):
//   FIXEDVSRANDOM  — must be defined
//   BINARYTRACES   — recommended
//   DIFFTRACELENGTH must NOT be defined (all traces are the same length;
//   KYBER_N calls to montgomery_reduce is a fixed-size loop)
//
// Build (add to existing compile_elmo.sh, no new source files needed —
// reduce.c is already linked):
//   arm-none-eabi-gcc -Os -mthumb -mcpu=cortex-m0 -I. -I./common -c \
//       elmo_montgomery_FvR.c
//   # Then link as normal with elmoasmfunctions.s etc.
//
// Results:
//   output/fixedvsrandomtstatistics.txt  — t-statistic per cycle
//   output/asmoutput/asmtrace00001.txt   — cross-reference for leaky cycles
//   output/nonprofiledindexes/indextrace00001.txt — non-profiled count

#include <stdint.h>
#include <stddef.h>
#include "reduce.h"    /* int16_t montgomery_reduce(int32_t a)   */
#include "params.h"    /* KYBER_N = 256, KYBER_Q = 3329          */

/* ------------------------------------------------------------------ */
/* ELMO triggers and helpers                                           */
/* ------------------------------------------------------------------ */
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

/* Number of traces per set.  Total = 2 * NOTRACES = 2000.
 * Increase to 5000 for higher statistical power if simulation time
 * allows.                                                            */
#define NOTRACES 1000

/* Fixed input to montgomery_reduce.
 *
 * (int32_t)KYBER_Q = 3329, chosen because:
 *   - Non-zero and non-trivial: avoids degenerate cases where the
 *     intermediate u = (int16_t)(a * QINV) collapses to 0, which
 *     would suppress observable power variation in the multiply step.
 *   - Well within |a| ≤ q*2^15: no risk of undefined behaviour.
 *   - Produces a non-zero output (montgomery_reduce(q) = 1 in exact
 *     arithmetic), exercising the full reduction path.
 *
 * Alternative: (int32_t)1664 * 1664 = 2,768,896 better mimics a real
 * fqmul product of two mid-range NTT coefficients.                   */
//#define FIXED_INPUT ((int32_t)KYBER_Q)
#define FIXED_INPUT ((int32_t)1664 * 1664)

/* ------------------------------------------------------------------ */
/* Static allocation — avoids stack overflow on M0's limited stack    */
/* ------------------------------------------------------------------ */

/* Pre-computed random inputs for the random set.
 * Filled by fill_random_inputs() BEFORE each trigger so that randbyte
 * activity stays outside the measured trace window.                  */
static int32_t rand_buf[KYBER_N];

/* Output sink — results from montgomery_reduce are written here.
 * Using a static (non-local) destination prevents the compiler from
 * eliminating the reduction calls as dead code under -Os.            */
static int16_t out_buf[KYBER_N];

/* ------------------------------------------------------------------ */
/* Random input generator (called outside the trigger)                */
/* ------------------------------------------------------------------ */
/*
 * Fills rand_buf with KYBER_N fresh random int32_t values.
 *
 * Three randbyte calls per entry produce a 24-bit value (b2:b1:b0)
 * in [0, 16,777,215], always satisfying |a| ≤ q*2^15 = 109,166,592.
 *
 * The outer for-loop has a backward branch but all randbyte calls are
 * in straight-line code within the loop body — the same structure as
 * the working CBD sampler in elmo_NTT_FvR.c.  No randbyte call
 * appears inside a condition-tested loop (the pattern that caused
 * ELMO to stall at the fixed/random boundary in earlier attempts).
 */
static void fill_random_inputs(void) {
    int i;
    for (i = 0; i < KYBER_N; i++) {
        uint8_t b0, b1, b2;
        randbyte(&b0);
        randbyte(&b1);
        randbyte(&b2);
        rand_buf[i] = (int32_t)(  ((uint32_t)b2 << 16)
                                | ((uint32_t)b1 <<  8)
                                |  (uint32_t)b0       );
    }
}

static void register_wash(void) {
    // Clear common registers to a constant state
    __asm__ volatile (
        "mov r0, #0\n\t" "mov r1, #0\n\t" "mov r2, #0\n\t"
        "mov r3, #0\n\t" "mov r4, #0\n\t" "mov r5, #0\n\t"
        "mov r6, #0\n\t" "mov r7, #0\n"
        ::: "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"
    );
}

/* ------------------------------------------------------------------ */
/* Main: fixed set first, then random set (required by ELMO FvR)      */
/* ------------------------------------------------------------------ */
int main(void) {

    int i, j;

    /* ---- FIXED set ---- */
    /* Every trace calls PQCLEAN_MLKEM512_CLEAN_montgomery_reduce(FIXED_INPUT) 256 times.
     * Inputs are identical across all 1000 traces → fixed set.       */
    for (i = 0; i < NOTRACES; i++) {
        register_wash();
        starttrigger();
        for (j = 0; j < KYBER_N; j++) {
            out_buf[j] = PQCLEAN_MLKEM512_CLEAN_montgomery_reduce(FIXED_INPUT);
        }
        endtrigger();
    }

    /* ---- RANDOM set ---- */
    /* Each trace pre-fills rand_buf with fresh 24-bit random inputs
     * (outside the trigger), then calls montgomery_reduce on each
     * entry inside the trigger.                                       */
    for (i = 0; i < NOTRACES; i++) {
        fill_random_inputs();       /* randbyte calls — outside trigger */
        register_wash();
        starttrigger();
        for (j = 0; j < KYBER_N; j++) {
            out_buf[j] = PQCLEAN_MLKEM512_CLEAN_montgomery_reduce(rand_buf[j]);
        }
        endtrigger();
    }

    /* Signal ELMO to run the Welch t-test.
     * Any cycle with |t| > 4.5 flagged as leaking (99.999 % CI).
     * Cross-reference fixedvsrandomtstatistics.txt line N with
     * asmtrace00001.txt line N to identify the leaking instruction.  */
    endprogram();
    return 0;
}
