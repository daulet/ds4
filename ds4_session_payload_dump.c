#include "ds4.h"

#include <locale.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *fp) {
    fputs("usage: ds4-session-payload-dump [--graph-plan|--graph-probe] [output.json]\n", fp);
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "C");
    bool graph_plan = false;
    bool graph_probe = false;
    const char *output_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--graph-plan")) {
            if (graph_probe) {
                usage(stderr);
                return 2;
            }
            graph_plan = true;
        } else if (!strcmp(argv[i], "--graph-probe")) {
            if (graph_plan) {
                usage(stderr);
                return 2;
            }
            graph_probe = true;
        } else if (!output_path) {
            output_path = argv[i];
        } else {
            usage(stderr);
            return 2;
        }
    }

    FILE *fp = stdout;
    if (output_path) {
        fp = fopen(output_path, "wb");
        if (!fp) {
            perror("ds4-session-payload-dump");
            return 1;
        }
    }

    const int rc = graph_plan ?
        ds4_dump_graph_session_payload_plan_json(fp) :
        graph_probe ?
            ds4_dump_graph_session_payload_probe_json(fp) :
            ds4_dump_session_payload_shape_json(fp);
    if (fp != stdout && fclose(fp) != 0) {
        perror("ds4-session-payload-dump");
        return 1;
    }
    if (rc != 0) {
        fputs("ds4-session-payload-dump: failed to write payload oracle\n", stderr);
        return 1;
    }
    if (!output_path) fputc('\n', stdout);
    return 0;
}
