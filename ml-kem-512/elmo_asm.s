.syntax unified
.cpu cortex-m0
.thumb

.global _start
.global starttrigger
.global endtrigger
.global endprogram
.global readbyte
.global printbyte

_start:
    bl main       @ Jump to your C main function
    b endprogram  @ If main returns, stop

starttrigger:
    bx lr
endtrigger:
    bx lr
endprogram:
    bx lr
readbyte:
    bx lr
printbyte:
    bx lr
