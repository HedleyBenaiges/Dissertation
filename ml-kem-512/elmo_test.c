// Fixed vs Random TVLA for ML-KEM-512 Decapsulation
//
// !! The previous version hung because crypto_kem_keypair() and
//    crypto_kem_enc() invoke SHAKE-128 (Keccak) for matrix generation,
//    costing millions of M0 cycles before the first trigger ever fires.
//
//    Fix: fill sk/ct buffers directly with randbyte().  The bytes are
//    not a "real" keypair, but every arithmetic operation still executes
//    identically — coefficient values are just interpreted from the raw
//    bytes via polyvec_frombytes / poly_frombytes.  The power trace is
//    therefore valid for side-channel evaluation.
//
// Two targets (pick one with the #define below):
//
//   TEST_INDCPA_DEC  (default — recommended first)
//     Tests indcpa_dec only: polyvec_basemul with secret key,
//     montgomery_reduce, barrett_reduce, poly_invntt, poly_tomsg.
//     NO Keccak at all → each trace is fast (~NTT-sized).
//     This is where the highest-priority leakage lives.
//
//   TEST_FULL_DECAPS
//     Tests the complete crypto_kem_dec: everything above PLUS
//     re-encryption (gen_matrix → SHAKE), verify, and cmov.
//     Each trace will be MUCH slower due to Keccak inside indcpa_enc.
//     Use small NOTRACES (e.g. 100) and be patient.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "params.h"
#include "indcpa.h"    /* PQCLEAN_MLKEM512_CLEAN_indcpa_dec            */
#include "api.h"       /* crypto_kem_dec, size macros                  */

/* ------------------------------------------------------------------ */
/* Choose ONE test target                                              */
/* ------------------------------------------------------------------ */
#define TEST_INDCPA_DEC          /* fast: polynomial arithmetic only   */
/* #define TEST_FULL_DECAPS */   /* slow: full decaps incl. SHAKE      */

/* ------------------------------------------------------------------ */
/* ELMO hardware-function interface                                   */
/* ------------------------------------------------------------------ */
#define starttrigger() (*((volatile uint32_t *)0xE0000004) = 1)
#define endtrigger()   (*((volatile uint32_t *)0xE0000004) = 0)
#define endprogram()   (*((volatile uint32_t *)0xF0000000) = 0)
#define randbyte(addr) (*(addr) = (uint8_t)*((volatile uint32_t *)0xE1000004))

/* ------------------------------------------------------------------ */
/* Trace count                                                         */
/* ------------------------------------------------------------------ */
#ifdef TEST_INDCPA_DEC
#define NOTRACES 1000      /* fast traces — use large N                */
#else
#define NOTRACES 100       /* full decaps is slow — start small        */
#endif

/* ------------------------------------------------------------------ */
/* Buffer sizes                                                        */
/* ------------------------------------------------------------------ */

/* indcpa_dec secret key = serialised s vector = 12 * k * n/4 bytes   */
/* For k=2, n=256: 12 * 2 * 64 = 1536 … but PQClean uses             */
/* KYBER_INDCPA_SECRETKEYBYTES which equals KYBER_POLYVECBYTES = 768  */
#define INDCPA_SK_BYTES  KYBER_POLYVECBYTES          /* 768           */
#define INDCPA_CT_BYTES  KYBER_INDCPA_BYTES          /* 768           */
#define INDCPA_MSG_BYTES KYBER_INDCPA_MSGBYTES       /* 32            */

/* Full KEM sizes                                                      */
#define SK_BYTES PQCLEAN_MLKEM512_CLEAN_CRYPTO_SECRETKEYBYTES  /* 1632 */
#define CT_BYTES PQCLEAN_MLKEM512_CLEAN_CRYPTO_CIPHERTEXTBYTES /* 768  */
#define SS_BYTES PQCLEAN_MLKEM512_CLEAN_CRYPTO_BYTES           /* 32   */

/* ------------------------------------------------------------------ */
/* Static buffers                                                      */
/* ------------------------------------------------------------------ */
#ifdef TEST_INDCPA_DEC
static uint8_t sk_cpa[INDCPA_SK_BYTES];      /* indcpa secret key     */
static uint8_t ct_fixed[INDCPA_CT_BYTES];     /* fixed ciphertext      */
static uint8_t ct[INDCPA_CT_BYTES];           /* working ct buffer     */
static uint8_t msg[INDCPA_MSG_BYTES];         /* decoded message out   */
#else
static uint8_t sk[SK_BYTES];                  /* full KEM secret key   */
static uint8_t ct_fixed[CT_BYTES];            /* fixed ciphertext      */
static uint8_t ct[CT_BYTES];                  /* working ct buffer     */
static uint8_t ss[SS_BYTES];                  /* shared-secret output  */
#endif

/* ------------------------------------------------------------------ */
/* randombytes — PQClean's randomness hook, routed to ELMO             */
/* (Only needed for TEST_FULL_DECAPS: indcpa_enc inside decaps uses it)*/
/* ------------------------------------------------------------------ */
void randombytes(uint8_t *buf, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) {
        randbyte(&buf[i]);
    }
}

/* ------------------------------------------------------------------ */
/* Fill a buffer with random bytes from ELMO's RNG                     */
/* ------------------------------------------------------------------ */
static void fill_random(uint8_t *buf, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) {
        randbyte(&buf[i]);
    }
}

void *memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = dst;
    const uint8_t *s = src;
    size_t i;
    for (i = 0; i < n; i++) { d[i] = s[i]; }
    return dst;
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

int main(void) {

    int i;

    /* ============================================================== */
    /* SETUP — fill buffers with randbyte, NO expensive SHAKE calls    */
    /* ============================================================== */

#ifdef TEST_INDCPA_DEC
    /*
     * indcpa_dec(msg, ct, sk_cpa) does:
     *   1. polyvec_frombytes(&b, ct)         — deserialise ct vector
     *   2. poly_frombytes(&v, ct+vecbytes)   — deserialise ct poly
     *   3. polyvec_ntt(&b)                   — NTT of ciphertext
     *   4. polyvec_basemul_acc_montgomery    — SECRET KEY MULTIPLY ←←←
     *   5. poly_invntt_tomont               — inverse NTT
     *   6. poly_sub, poly_reduce            — subtract & reduce
     *   7. poly_tomsg(msg, &mp)             — message decoding
     *
     * All polynomial arithmetic, zero SHAKE.  This is the fastest
     * test and covers the most exploitable leakage surface.
     */

    /* Random CPA secret key — polyvec_frombytes will parse these     */
    /* bytes into polynomial coefficients; values are arbitrary but   */
    /* non-trivial, exercising the full dynamic range of the multiply.*/
    fill_random(sk_cpa, INDCPA_SK_BYTES);

    /* Fixed ciphertext — one random snapshot, reused for fixed set   */
    fill_random(ct_fixed, INDCPA_CT_BYTES);

    /* ---- FIXED set ---- */
    for (i = 0; i < NOTRACES; i++) {
        memcpy(ct, ct_fixed, INDCPA_CT_BYTES);
        register_wash();
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_indcpa_dec(msg, ct, sk_cpa);
        endtrigger();
    }

    /* ---- RANDOM set ---- */
    for (i = 0; i < NOTRACES; i++) {
        fill_random(ct, INDCPA_CT_BYTES);
        register_wash();
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_indcpa_dec(msg, ct, sk_cpa);
        endtrigger();
    }

#else /* TEST_FULL_DECAPS */
    /*
     * Full crypto_kem_dec includes indcpa_dec PLUS:
     *   - SHA3-512 hash
     *   - indcpa_enc (re-encryption — calls gen_matrix → SHAKE-128)
     *   - verify (constant-time comparison)
     *   - cmov  (implicit rejection)
     *   - SHA3-256 final hash
     *
     * WARNING: each trace takes a very long time due to Keccak.
     * The sk buffer must be the full 1632-byte KEM secret key:
     *   sk = sk_cpa (768) || pk (800) || H(pk) (32) || z (32)
     * Filling with random bytes is fine — every function still
     * executes, the output is just cryptographic garbage.
     */

    fill_random(sk, SK_BYTES);
    fill_random(ct_fixed, CT_BYTES);

    /* ---- FIXED set ---- */
    for (i = 0; i < NOTRACES; i++) {
        memcpy(ct, ct_fixed, CT_BYTES);
        register_wash();
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_crypto_kem_dec(ss, ct, sk);
        endtrigger();
    }

    /* ---- RANDOM set ---- */
    for (i = 0; i < NOTRACES; i++) {
        fill_random(ct, CT_BYTES);
        register_wash();
        starttrigger();
        PQCLEAN_MLKEM512_CLEAN_crypto_kem_dec(ss, ct, sk);
        endtrigger();
    }

#endif

    endprogram();
    return 0;
}
