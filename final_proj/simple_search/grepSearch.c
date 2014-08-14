#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

int main(void) {
    struct timeval start, end, diff;
    int bOk;

    gettimeofday(&start, 0);
    bOk = system("grep -cw hello ../DATA/UnicodeSample.txt > /dev/null");
    gettimeofday(&end, 0);
    long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
    printf("Elapsed time: %lld\n", elapsed);
    return 0;
}

