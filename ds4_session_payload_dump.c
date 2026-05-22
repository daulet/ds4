#include "ds4.h"

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>

static void usage(FILE *fp) {
    fputs("usage: ds4-session-payload-dump [output.json]\n", fp);
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "C");
    if (argc > 2) {
        usage(stderr);
        return 2;
    }

    FILE *fp = stdout;
    if (argc == 2) {
        fp = fopen(argv[1], "wb");
        if (!fp) {
            perror("ds4-session-payload-dump");
            return 1;
        }
    }

    const int rc = ds4_dump_session_payload_shape_json(fp);
    if (fp != stdout && fclose(fp) != 0) {
        perror("ds4-session-payload-dump");
        return 1;
    }
    if (rc != 0) {
        fputs("ds4-session-payload-dump: failed to write payload oracle\n", stderr);
        return 1;
    }
    if (argc != 2) fputc('\n', stdout);
    return 0;
}
