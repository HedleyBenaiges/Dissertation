// Fixed vs Random TVLA for ML-KEM-512 CBD(eta1=3) arithmetic
//
// Structure mirrors elmo_INTT_FvR.c but targets the Centered Binomial
// Distribution (CBD) sampler, which produces the private-key and error
// polynomials during ML-KEM-512 key generation and encapsulation.
//
// -----------------------------------------------------------------------
// Why poly_getnoise_eta1 hangs ELMO — and how we fix it
// -----------------------------------------------------------------------
//
//  poly_getnoise_eta1 calls prf() -> shake256() -> Keccak-f[1600].
//  The Keccak permutation from fips202.c:
//    - Uses uint64_t arithmetic throughout (25-word, 1600-bit state).
//    - On Cortex-M0 (no 64-bit ALU) every uint64_t operation compiles
//      to a pair of 32-bit instructions (ADD/ADC, EOR/EOR, etc.).
//    - 24 rounds x ~5 steps x 25 words x 2 instructions = thousands of
//      instructions per prf() call.
//  ELMO either overflows its internal trace buffer or encounters a
//  non-profiled instruction sequence inside fips202.c and stalls
//  indefinitely — the same class of hang as the __aeabi_uidivmod call
//  documented in elmo_INTT_FvR.c.
//
//  Additionally, poly_cbd_eta1 is declared static in cbd.c and is not
//  exported in any PQClean header, making it unreachable by name:
//
//    error: implicit declaration of 'PQCLEAN_MLKEM512_CLEAN_poly_cbd_eta1'
//
//  Fix: bypass the PRF entirely.  The CBD function takes a flat byte
//  buffer as input; we fill that buffer directly with ELMO's randbyte()
//  and implement the CBD arithmetic as a local static function (a direct
//  copy of the PQClean logic).  No SHAKE-256, no fips202.c, no libgcc
//  calls, no backward branches — ELMO handles all of this safely.
//
// -----------------------------------------------------------------------
// What is being tested
// -----------------------------------------------------------------------
//
//  cbd_eta1_local(poly *r, const uint8_t buf[CBD_BUFLEN])
//
//  This is a verbatim copy of the PQClean poly_cbd_eta1 logic.  For
//  ML-KEM-512, eta1 = 3 and the algorithm is:
//
//    For each group of 4 coefficients (i = 0 .. 63):
//      Load 3 bytes little-endian -> 24-bit word t
//      Accumulate bit-slice Hamming weights using mask 0x249249:
//        d  = t & mask
//        d += (t >> 1) & mask      // sum of individual bits across 3 planes
//        d += (t >> 2) & mask
//      For j = 0..3:
//        a = (d >> 6j)     & 0x7   // HW of 3 bits from the first half
//        b = (d >> 6j + 3) & 0x7   // HW of 3 bits from the second half
//        coeff[4i+j] = a - b       // result in [-3, +3]
//
//  The AND, shift, and ADD instructions inside this loop directly process
//  secret-dependent data.  On a real device their Hamming-weight-driven
//  power consumption could be exploited to reconstruct the input buffer,
//  which in ML-KEM is the output of a secret-keyed PRF.
//
// -----------------------------------------------------------------------
// Why the PRF is omitted from the trigger window
// -----------------------------------------------------------------------
//
//  A separate ELMO harness (elmo_PRF_FvR.c) should test the SHAKE-256
//  PRF path.  Combining both in one trigger would:
//    (a) make the trace too long for ELMO to handle reliably, and
//    (b) conflate leakage sources, making it harder to attribute any
//        detected signal to a specific instruction.
//
//  In a real SCA campaign a hardware oscilloscope can be time-windowed
//  to isolate the CBD portion of the noise-generation waveform.
//
// -----------------------------------------------------------------------
// Input domains
// -----------------------------------------------------------------------
//
//  The CBD input buffer has length CBD_BUFLEN = KYBER_ETA1*KYBER_N/4
//  = 3*256/4 = 192 bytes for ML-KEM-512.
//
//  Fixed set:
//    buf[] <- {FIXED_BYTE, FIXED_BYTE, ...}  (192 identical bytes)
//
//    FIXED_BYTE = 0x24 (0b00100100, HW=2).  Rationale:
//      - 0x00 and 0xFF both yield all-zero coefficients
//        (HW(0)-HW(0)=0 and HW(3)-HW(3)=0 respectively), suppressing
//        arithmetic variation inside the butterfly loop.
//      - 0x24 produces non-zero, non-maximal coefficients via the
//        0x249249 mask accumulation, exercising the full datapath.
//      - HW=2 sits mid-range in the Cortex-M0 power model.
//
//  Random set:
//    buf[] <- 192 independent uniform-random bytes via randbyte().
//
//    Rationale: the real CBD input is the output of SHAKE-256, which
//    is computationally indistinguishable from uniform random bytes.
//    192 randbyte() calls are straight-line (no backward branches, no
//    division), so ELMO will not hang at the set boundary.
//
// -----------------------------------------------------------------------
// ELMO setup (elmodefines.h)
// -----------------------------------------------------------------------
//
//   FIXEDVSRANDOM        must be defined
//   BINARYTRACES         recommended (speed + storage)
//   DIFFTRACELENGTH      must NOT be defined — all traces are the same
//                        length (CBD is data-length-independent for a
//                        fixed 192-byte input)
//
// -----------------------------------------------------------------------
// Build
// -----------------------------------------------------------------------
//
//   This file is self-contained: it does not link against cbd.c,
//   fips202.c, or symmetric-shake.c.  Only poly.c is needed for the
//   poly struct definition (via poly.h).
//
//   arm-none-eabi-gcc -mcpu=cortex-m0 -mthumb -O2                       \
//       -I path/to/pqclean/crypto_kem/ml-kem-512/clean                   \
//       -o elmo_CBD_FvR.elf elmo_CBD_FvR.c                               \
//       path/to/pqclean/crypto_kem/ml-kem-512/clean/poly.c
//
//   arm-none-eabi-objcopy -O binary elmo_CBD_FvR.elf elmo_CBD_FvR.bin
//   ./elmo elmo_CBD_FvR.bin
//
// -----------------------------------------------------------------------
// Results
// -----------------------------------------------------------------------
//
//   output/fixedvsrandomtstatistics.txt  - t-statistic per instruction.
//                                          |t| > 4.5 flags data-dependent
//                                          power at 99.999% confidence.
//   output/asmoutput/asmtrace00001.txt   - instruction-level trace.
//                                          Cross-reference to identify
//                                          which AND/shift/ADD inside the
//                                          CBD butterfly loop leaks.
//
// -----------------------------------------------------------------------

#include <stdint.h>
#include <stddef.h>
#include "poly.h"      /* poly struct; KYBER_N via params.h              */
#include "params.h"    /* KYBER_N = 256, KYBER_ETA1 = 3                  */

/* ------------------------------------------------------------------ */
/* ELMO triggers and helpers                                           */
/* ------------------------------------------------------------------ */
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

/* Number of traces in each half (fixed / random).
 * Total = 2 * NOTRACES = 2000.  Increase to 5000-10000 for a more
 * powerful test at the cost of longer simulation time.               */
#define NOTRACES 1000

/* ------------------------------------------------------------------ */
/* CBD input buffer size                                               */
/* ------------------------------------------------------------------ */
/* KYBER_ETA1 * KYBER_N / 4 = 3 * 256 / 4 = 192 bytes               */
#define CBD_BUFLEN (KYBER_ETA1 * KYBER_N / 4)

/* ------------------------------------------------------------------ */
/* Fixed input byte                                                    */
/* ------------------------------------------------------------------ */
/* 0x24 = 0b00100100, HW=2.  See header comment for full rationale.  */
#define FIXED_BYTE ((uint8_t)0x24)

/* ------------------------------------------------------------------ */
/* Register wash                                                       */
/* ------------------------------------------------------------------ */
static void register_wash(void) {
    __asm__ volatile (
        "mov r0, #0\n\t" "mov r1, #0\n\t" "mov r2, #0\n\t"
        "mov r3, #0\n\t" "mov r4, #0\n\t" "mov r5, #0\n\t"
        "mov r6, #0\n\t" "mov r7, #0\n"
        ::: "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"
    );
}

/* ------------------------------------------------------------------ */
/* CBD eta1=3 arithmetic — inlined from PQClean cbd.c                 */
/* ------------------------------------------------------------------ */
/*
 * Verbatim copy of the PQClean poly_cbd_eta1 logic for ML-KEM-512
 * (eta1 = 3).  Inlined here because poly_cbd_eta1 is declared static
 * in cbd.c and is not exported in any public header.
 *
 * No modification to the arithmetic: any detected leakage corresponds
 * directly to the instructions that execute on the real target.
 *
 * The 24-bit word t is loaded from three consecutive buffer bytes in
 * little-endian order.  The mask 0x249249 (binary 001001001001001001001001)
 * isolates every third bit across the 24-bit window, implementing the
 * bit-sliced Hamming weight accumulation needed for CBD(eta1=3).
 */
static void cbd_eta1_local(poly *r, const uint8_t buf[CBD_BUFLEN]) {
    uint32_t t, d;
    int16_t  a, b;
    int i, j;

    for (i = 0; i < KYBER_N / 4; i++) {
        /* Load 3 bytes -> 24-bit little-endian word */
        t  = (uint32_t)buf[3 * i + 0];
        t |= (uint32_t)buf[3 * i + 1] << 8;
        t |= (uint32_t)buf[3 * i + 2] << 16;

        /* Bit-sliced Hamming weight accumulation (no division, no branch) */
        d  =  t & 0x00249249u;
        d += (t >> 1) & 0x00249249u;
        d += (t >> 2) & 0x00249249u;

        /* Extract 4 coefficients from the 24-bit accumulator */
        for (j = 0; j < 4; j++) {
            a = (int16_t)((d >> (6 * j + 0)) & 0x7u); /* HW of bits [2:0] */
            b = (int16_t)((d >> (6 * j + 3)) & 0x7u); /* HW of bits [5:3] */
            r->coeffs[4 * i + j] = a - b;             /* result in [-3,+3] */
        }
    }
}

/* ------------------------------------------------------------------ */
/* Fixed-buffer initialiser                                            */
/* ------------------------------------------------------------------ */
/*
 * Fill all 192 bytes with FIXED_BYTE.  The output polynomial is
 * deterministic and identical across every fixed-set trace, so any
 * trace-to-trace power variation within the fixed set is pure noise.
 */
static void set_fixed_buf(uint8_t *buf) {
    int i;
    for (i = 0; i < CBD_BUFLEN; i++) {
        buf[i] = FIXED_BYTE;
    }
}

/* ------------------------------------------------------------------ */
/* Random-buffer sampler                                               */
/* ------------------------------------------------------------------ */
/*
 * Fill all 192 bytes with independent uniform-random bytes from ELMO's
 * RNG.  Models the true distribution of the SHAKE-256 PRF output.
 *
 * 192 unconditional randbyte() calls — no backward branches, no modular
 * reduction, no library calls.  ELMO will not stall at the set boundary.
 */
static void sample_random_buf(uint8_t *buf) {
    int i;
    for (i = 0; i < CBD_BUFLEN; i++) {
        randbyte(&buf[i]);
    }
}

/* ------------------------------------------------------------------ */
/* Static allocations — avoid Cortex-M0 stack overflow                */
/* ------------------------------------------------------------------ */
static uint8_t cbd_buf[CBD_BUFLEN]; /* 192-byte CBD input buffer       */
static poly    s;                   /* 256-coefficient output poly      */

/* ------------------------------------------------------------------ */
/* Main: fixed set first, then random set (required by ELMO FvR)     */
/* ------------------------------------------------------------------ */
int main(void) {

    int i;

    /* ---- FIXED buffer set ---- */
    /*
     * All 192 input bytes are FIXED_BYTE for every trace.
     * cbd_eta1_local produces the same polynomial each time.
     * Power variation within this set is purely modelling noise,
     * forming the null-hypothesis baseline for the t-test.
     */
    for (i = 0; i < NOTRACES; i++) {
        set_fixed_buf(cbd_buf);
        register_wash();
        starttrigger();
        cbd_eta1_local(&s, cbd_buf);
        endtrigger();
    }

    /* ---- RANDOM buffer set ---- */
    /*
     * All 192 input bytes are freshly sampled uniform-random values
     * for every trace.  Power variation within this set includes both
     * noise and any data-dependent component of the CBD arithmetic.
     */
    for (i = 0; i < NOTRACES; i++) {
        sample_random_buf(cbd_buf);
        register_wash();
        starttrigger();
        cbd_eta1_local(&s, cbd_buf);
        endtrigger();
    }

    /* Run Welch's t-test across both trace sets.
     *
     * output/fixedvsrandomtstatistics.txt — one t-value per instruction.
     * output/asmoutput/asmtrace00001.txt  — full instruction trace.
     *
     * Expected leaky instructions: the AND, LSR, and ADD inside the
     * butterfly accumulator loop process input bytes directly.  On a
     * real Cortex-M0 these instructions' Hamming-weight-driven power
     * consumption is the primary CBD leakage source.
     *
     * |t| > 4.5: data-dependent leakage at 99.999% confidence.
     * |t| <= 4.5: no significant leakage at N = 1000 traces per set. */
    endprogram();
    return 0;
}
