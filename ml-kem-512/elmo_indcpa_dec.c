// DPA Trace Generation for CPA Attack on ML-KEM-512
//
// Methodology: Mujdei et al., "Side-Channel Analysis of Lattice-Based
// Post-Quantum Cryptography: Exploiting Polynomial Multiplication"
// Section 4.3 — CPA on NTT-based multiplication (Kyber).
//
// Target: polyvec_basemul_acc_montgomery inside indcpa_dec.
//   - ML-KEM-512 uses a 7-layer incomplete NTT with schoolbook threshold 2.
//   - The basemul computes 128 two-coefficient multiplications in NTT domain.
//   - Each basemul stores r[0], r[1] via strh — profiled by ELMO (strh is in
//     the profiled instruction set) and leaks HW(r[0]).
//
// Strategy:
//   - Secret key sk_cpa is FIXED across all traces.
//   - Ciphertext ct is RANDOM and DIFFERENT every trace.
//   - ELMO's randbyte() records all random bytes to randdata.txt.
//     - First  INDCPA_SK_BYTES  bytes  = the fixed sk_cpa.
//     - Next   NOTRACES * INDCPA_CT_BYTES  bytes = ct for trace 1, 2, ...
//   - The Python CPA script reads randdata.txt to reconstruct ct[i], 
//     computes the NTT-domain ciphertext coefficients, and correlates
//     hypothetical HW(basemul result) with ELMO trace power at each cycle.
//
// Build:  use compile_elmo_profiled.sh (patched binary reduces non-profiled
//         instructions; strh in basemul must be profiled for CPA to work).
//         Do NOT define FIXEDVSRANDOM in elmodefines.h — this is DPA mode.
//
// Run:    ./elmo elmo_target.bin
//         Output: output/trace00001.txt ... output/traceNNNNN.txt
//                 output/randdata.txt  (ct bytes for Python)
//                 output/asmoutput/asmtrace00001.txt  (for cycle identification)

#include <stdint.h>
#include <stddef.h>
#include "params.h"
#include "indcpa.h"

/* ------------------------------------------------------------------ */
/* ELMO hardware-function interface                                   */
/* ------------------------------------------------------------------ */
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

/* ------------------------------------------------------------------ */
/* Trace count                                                         */
/* The paper (Section 5.3) uses 200 traces for Kyber.  500 gives more */
/* statistical power at the cost of longer runtime.                   */
/* ------------------------------------------------------------------ */
#define NOTRACES 1000

/* ------------------------------------------------------------------ */
/* Buffer sizes                                                        */
/* INDCPA_SK_BYTES = KYBER_POLYVECBYTES = k * 12 * n/8 = 2*384 = 768  */
/* INDCPA_CT_BYTES = KYBER_INDCPA_BYTES = 768 (640 polyvec + 128 poly) */
/* INDCPA_MSG_BYTES = KYBER_INDCPA_MSGBYTES = 32                       */
/* ------------------------------------------------------------------ */
#define INDCPA_SK_BYTES  KYBER_POLYVECBYTES
#define INDCPA_CT_BYTES  KYBER_INDCPA_BYTES
#define INDCPA_MSG_BYTES KYBER_INDCPA_MSGBYTES

/* ------------------------------------------------------------------ */
/* Static buffers — avoids stack overflow on M0's 4 KB stack           */
/* ------------------------------------------------------------------ */
static uint8_t sk_cpa[INDCPA_SK_BYTES];
static uint8_t ct[INDCPA_CT_BYTES];
static uint8_t msg[INDCPA_MSG_BYTES];

/* ------------------------------------------------------------------ */
/* randombytes / PQCLEAN_randombytes — linker stubs                   */
/* ------------------------------------------------------------------ */
void randombytes(uint8_t *buf, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) { randbyte(&buf[i]); }
}
void PQCLEAN_randombytes(uint8_t *buf, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) { randbyte(&buf[i]); }
}

/* ------------------------------------------------------------------ */
/* Stdlib-free helpers                                                 */
/* ------------------------------------------------------------------ */
static void fill_random(uint8_t *buf, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) { randbyte(&buf[i]); }
}

/* Optional: our own memcpy so PQClean's internal calls stay profiled  */
void *memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = dst;
    const uint8_t *s = src;
    size_t i;
    for (i = 0; i < n; i++) { d[i] = s[i]; }
    return dst;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {
    int i;

    /* ============================================================== */
    /* SETUP: generate fixed secret key                               */
    /* fill_random → randbyte → recorded in randdata.txt              */
    /* Python reads first INDCPA_SK_BYTES bytes to reconstruct sk_cpa  */
    /* ============================================================== */
    fill_random(sk_cpa, INDCPA_SK_BYTES);

    /* ============================================================== */
    /* TRACE LOOP: fixed key, random ciphertext per trace             */
    /* ============================================================== */
    for (i = 0; i < NOTRACES; i++) {
        /* Each fill_random call appends INDCPA_CT_BYTES bytes to     */
        /* randdata.txt.  Python reads trace i's ct as bytes          */
        /* [INDCPA_SK_BYTES + i*INDCPA_CT_BYTES :                     */
        /*  INDCPA_SK_BYTES + (i+1)*INDCPA_CT_BYTES].                 */
        fill_random(ct, INDCPA_CT_BYTES);

        /* Trigger: ELMO records power from here to endtrigger().      */
        /* Target region: polyvec_basemul_acc_montgomery inside        */
        /* indcpa_dec.  The basemul stores coefficients via strh,      */
        /* which is profiled by ELMO and leaks Hamming weight.        */
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_indcpa_dec(msg, ct, sk_cpa);
        endtrigger();
    }

    endprogram();
    return 0;
}
