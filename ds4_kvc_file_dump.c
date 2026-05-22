#include "ds4_kvstore.h"

#include <errno.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    bool ok;
    ds4_kvstore_entry entry;
    uint64_t trailer_bytes;
    bool expected_total_ok;
} kvc_read_probe;

static void die(const char *msg) {
    fprintf(stderr, "ds4-kvc-file-dump: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void write_json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"': fputs("\\\"", fp); break;
        case '\\': fputs("\\\\", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
            if (c < 0x20) fprintf(fp, "\\u%04x", c);
            else fputc(c, fp);
            break;
        }
    }
    fputc('"', fp);
}

static void write_hex_string(FILE *fp, const uint8_t *bytes, size_t len) {
    static const char hex[] = "0123456789abcdef";
    fputc('"', fp);
    for (size_t i = 0; i < len; i++) {
        fputc(hex[bytes[i] >> 4], fp);
        fputc(hex[bytes[i] & 15], fp);
    }
    fputc('"', fp);
}

static bool checked_total(size_t text_len, size_t payload_len, size_t trailer_len,
                          size_t *total) {
    size_t fixed = DS4_KVSTORE_FIXED_HEADER + 4u;
    if (text_len > SIZE_MAX - fixed) return false;
    size_t n = fixed + text_len;
    if (payload_len > SIZE_MAX - n) return false;
    n += payload_len;
    if (trailer_len > SIZE_MAX - n) return false;
    *total = n + trailer_len;
    return true;
}

static uint8_t *make_kvc_bytes(uint8_t quant_bits,
                               uint8_t reason,
                               uint8_t ext_flags,
                               uint32_t tokens,
                               uint32_t hits,
                               uint32_t ctx_size,
                               uint64_t created_at,
                               uint64_t last_used,
                               const uint8_t *text,
                               size_t text_len,
                               const uint8_t *payload,
                               size_t payload_len,
                               const uint8_t *trailer,
                               size_t trailer_len,
                               size_t *len_out,
                               uint8_t header_out[DS4_KVSTORE_FIXED_HEADER],
                               uint8_t text_len_out[4]) {
    if (text_len > UINT32_MAX) die("text fixture too large");
    size_t total = 0;
    if (!checked_total(text_len, payload_len, trailer_len, &total)) {
        die("fixture size overflow");
    }

    uint8_t *bytes = xmalloc(total);
    ds4_kvstore_fill_header(header_out, quant_bits, reason, ext_flags, tokens,
                            hits, ctx_size, created_at, last_used,
                            (uint64_t)payload_len);
    ds4_kvstore_le_put32(text_len_out, (uint32_t)text_len);
    memcpy(bytes, header_out, DS4_KVSTORE_FIXED_HEADER);
    memcpy(bytes + DS4_KVSTORE_FIXED_HEADER, text_len_out, 4);
    memcpy(bytes + DS4_KVSTORE_FIXED_HEADER + 4, text, text_len);
    if (payload_len) {
        memcpy(bytes + DS4_KVSTORE_FIXED_HEADER + 4 + text_len,
               payload, payload_len);
    }
    if (trailer_len) {
        memcpy(bytes + DS4_KVSTORE_FIXED_HEADER + 4 + text_len + payload_len,
               trailer, trailer_len);
    }
    *len_out = total;
    return bytes;
}

static kvc_read_probe probe_bytes(const uint8_t *bytes,
                                  size_t len,
                                  const char sha[41],
                                  uint64_t expected_total) {
    kvc_read_probe probe = {0};
    char tmpl[256];
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0]) tmpdir = "/tmp";
    int nw = snprintf(tmpl, sizeof(tmpl), "%s/ds4-kvc-file-dump-XXXXXX", tmpdir);
    if (nw < 0 || (size_t)nw >= sizeof(tmpl)) die("temporary path too long");
    int fd = mkstemp(tmpl);
    if (fd < 0) die("mkstemp failed");
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        close(fd);
        unlink(tmpl);
        die("fdopen failed");
    }
    if (fwrite(bytes, 1, len, fp) != len || fclose(fp) != 0) {
        unlink(tmpl);
        die("failed to write temp KVC file");
    }

    ds4_kvstore_entry e = {0};
    probe.ok = ds4_kvstore_read_entry_file(tmpl, sha, &e);
    if (probe.ok) {
        uint64_t expected_without_trailer =
            DS4_KVSTORE_FIXED_HEADER + 4ull + e.text_bytes + e.payload_bytes;
        probe.trailer_bytes = e.file_size >= expected_without_trailer ?
                              e.file_size - expected_without_trailer : 0;
        probe.expected_total_ok = e.file_size == expected_total;
        probe.entry = e;
    }
    unlink(tmpl);
    return probe;
}

static void free_probe(kvc_read_probe *probe) {
    if (probe->ok) ds4_kvstore_entry_free(&probe->entry);
    memset(probe, 0, sizeof(*probe));
}

static void write_read_probe(FILE *fp, const kvc_read_probe *probe) {
    fprintf(fp, "{\"ok\":%s", probe->ok ? "true" : "false");
    if (probe->ok) {
        const ds4_kvstore_entry *e = &probe->entry;
        fprintf(fp,
                ",\"quant_bits\":%u,\"reason\":%u,\"ext_flags\":%u,"
                "\"tokens\":%u,\"hits\":%u,\"ctx_size\":%u,"
                "\"created_at\":%llu,\"last_used\":%llu,"
                "\"payload_bytes\":%llu,\"text_bytes\":%llu,"
                "\"file_size\":%llu,\"trailer_bytes\":%llu,"
                "\"expected_total_ok\":%s",
                e->quant_bits,
                e->reason,
                e->ext_flags,
                e->tokens,
                e->hits,
                e->ctx_size,
                (unsigned long long)e->created_at,
                (unsigned long long)e->last_used,
                (unsigned long long)e->payload_bytes,
                (unsigned long long)e->text_bytes,
                (unsigned long long)e->file_size,
                (unsigned long long)probe->trailer_bytes,
                probe->expected_total_ok ? "true" : "false");
    }
    fputc('}', fp);
}

static void write_budget(FILE *fp, size_t text_len, size_t payload_len,
                         size_t trailer_len, uint64_t budget_bytes) {
    ds4_kvstore kc = {0};
    kc.budget_bytes = budget_bytes;
    uint64_t file_bytes = 0;
    uint64_t required_bytes = 0;
    bool fits = ds4_kvstore_file_size_fits(&kc, (uint64_t)text_len,
                                           (uint64_t)payload_len,
                                           (uint64_t)trailer_len,
                                           &file_bytes, &required_bytes);
    fprintf(fp,
            "{\"budget_bytes\":%llu,\"fits\":%s,\"file_bytes\":%llu,"
            "\"required_bytes\":%llu}",
            (unsigned long long)budget_bytes,
            fits ? "true" : "false",
            (unsigned long long)file_bytes,
            (unsigned long long)required_bytes);
}

static void write_case(FILE *fp,
                       const char *name,
                       uint8_t quant_bits,
                       uint8_t reason,
                       uint8_t ext_flags,
                       uint32_t tokens,
                       uint32_t hits,
                       uint32_t ctx_size,
                       uint64_t created_at,
                       uint64_t last_used,
                       const uint8_t *text,
                       size_t text_len,
                       const uint8_t *payload,
                       size_t payload_len,
                       const uint8_t *trailer,
                       size_t trailer_len,
                       uint64_t budget_bytes,
                       bool first) {
    if (!first) fputs(",\n", fp);
    uint8_t header[DS4_KVSTORE_FIXED_HEADER];
    uint8_t text_len_bytes[4];
    size_t file_len = 0;
    uint8_t *bytes = make_kvc_bytes(quant_bits, reason, ext_flags, tokens,
                                    hits, ctx_size, created_at, last_used,
                                    text, text_len, payload, payload_len,
                                    trailer, trailer_len, &file_len,
                                    header, text_len_bytes);
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    kvc_read_probe probe = probe_bytes(bytes, file_len, sha, (uint64_t)file_len);

    fputs("    {\"name\":", fp);
    write_json_string(fp, name);
    fputs(",\"input\":{", fp);
    fprintf(fp,
            "\"quant_bits\":%u,\"reason\":%u,\"ext_flags\":%u,"
            "\"tokens\":%u,\"hits\":%u,\"ctx_size\":%u,"
            "\"created_at\":%llu,\"last_used\":%llu,",
            quant_bits, reason, ext_flags, tokens, hits, ctx_size,
            (unsigned long long)created_at,
            (unsigned long long)last_used);
    fputs("\"text_hex\":", fp);
    write_hex_string(fp, text, text_len);
    fputs(",\"payload_hex\":", fp);
    write_hex_string(fp, payload, payload_len);
    fputs(",\"trailer_hex\":", fp);
    write_hex_string(fp, trailer, trailer_len);
    fputs("},\"sha1\":", fp);
    write_json_string(fp, sha);
    fputs(",\"header_hex\":", fp);
    write_hex_string(fp, header, sizeof(header));
    fputs(",\"text_len_hex\":", fp);
    write_hex_string(fp, text_len_bytes, sizeof(text_len_bytes));
    fputs(",\"file_hex\":", fp);
    write_hex_string(fp, bytes, file_len);
    fprintf(fp,
            ",\"file_size\":%llu,\"expected_trailer_bytes\":%llu,"
            "\"budget\":",
            (unsigned long long)file_len,
            (unsigned long long)trailer_len);
    write_budget(fp, text_len, payload_len, trailer_len, budget_bytes);
    fputs(",\"read_entry\":", fp);
    write_read_probe(fp, &probe);
    fputc('}', fp);

    free_probe(&probe);
    free(bytes);
}

static uint8_t *copy_bytes(const uint8_t *bytes, size_t len) {
    uint8_t *copy = xmalloc(len);
    memcpy(copy, bytes, len);
    return copy;
}

static void put32le(uint8_t *p, uint32_t v) {
    ds4_kvstore_le_put32(p, v);
}

static void put64le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static void write_malformed(FILE *fp,
                            const char *name,
                            const char *expected_error,
                            const uint8_t *bytes,
                            size_t len,
                            const char sha[41],
                            uint64_t expected_total,
                            bool first) {
    if (!first) fputs(",\n", fp);
    kvc_read_probe probe = probe_bytes(bytes, len, sha, expected_total);
    fputs("    {\"name\":", fp);
    write_json_string(fp, name);
    fputs(",\"expected_error\":", fp);
    write_json_string(fp, expected_error);
    fputs(",\"file_hex\":", fp);
    write_hex_string(fp, bytes, len);
    fputs(",\"read_entry\":", fp);
    write_read_probe(fp, &probe);
    fputc('}', fp);
    free_probe(&probe);
}

static void write_cases(FILE *fp) {
    static const uint8_t text0[] = "Alpha beta gamma";
    static const uint8_t payload0[] = {0x00, 0x01, 0x02, 0x03, 0xff, 0x10, 0x20, 0x30};
    static const uint8_t text1[] = "<tool-visible-prefix>";
    static const uint8_t payload1[] = {0xaa, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44, 0x55};
    static const uint8_t trailer1[] = {'O', 'P', 'Q', 0x00, 0x01, 0x02};
    static const uint8_t text2[] = "visible transcript";
    static const uint8_t text3[] = "";
    static const uint8_t payload3[] = {0x42, 0x24};
    static const uint8_t trailer3[] = {'x', 'y', 'z'};

    fputs("  \"cases\": [\n", fp);
    write_case(fp, "plain_no_trailer", 2, DS4_KVSTORE_REASON_COLD, 0,
               600, 0, 32768, 1700000000ull, 1700000000ull,
               text0, sizeof(text0) - 1, payload0, sizeof(payload0),
               NULL, 0, 0, true);
    write_case(fp, "opaque_tool_trailer", 4, DS4_KVSTORE_REASON_CONTINUED,
               DS4_KVSTORE_EXT_TOOL_MAP, 2048, 7, 65536,
               1700000100ull, 1700000200ull,
               text1, sizeof(text1) - 1, payload1, sizeof(payload1),
               trailer1, sizeof(trailer1), 92, false);
    write_case(fp, "visible_flag_no_payload", 2, DS4_KVSTORE_REASON_AGENT_SESSION,
               DS4_KVSTORE_EXT_RESPONSES_VISIBLE, 700, 2, 32768,
               1700000300ull, 1700000400ull,
               text2, sizeof(text2) - 1, NULL, 0, NULL, 0, 70, false);
    write_case(fp, "empty_text_thinking_trailer", 2, DS4_KVSTORE_REASON_SHUTDOWN,
               DS4_KVSTORE_EXT_THINKING_VISIBLE, 1, 0, 4096,
               1700000500ull, 1700000501ull,
               text3, 0, payload3, sizeof(payload3),
               trailer3, sizeof(trailer3), 64, false);
    fputs("\n  ],\n", fp);
}

static void write_malformed_cases(FILE *fp) {
    static const uint8_t text[] = "Alpha beta gamma";
    static const uint8_t payload[] = {0x00, 0x01, 0x02, 0x03, 0xff, 0x10, 0x20, 0x30};
    static const uint8_t trailer[] = {'O', 'P', 'Q', 0x00, 0x01, 0x02};
    uint8_t header[DS4_KVSTORE_FIXED_HEADER];
    uint8_t text_len_bytes[4];
    size_t len = 0;
    uint8_t *base = make_kvc_bytes(2, DS4_KVSTORE_REASON_COLD, 0,
                                   600, 0, 32768, 1700000000ull,
                                   1700000000ull, text, sizeof(text) - 1,
                                   payload, sizeof(payload), trailer,
                                   sizeof(trailer), &len, header,
                                   text_len_bytes);
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, sizeof(text) - 1, sha);

    fputs("  \"malformed_cases\": [\n", fp);
    write_malformed(fp, "truncated_header", "truncated-header",
                    base, 20, sha, (uint64_t)len, true);

    uint8_t *bad = copy_bytes(base, len);
    bad[0] = 'X';
    write_malformed(fp, "bad_magic", "invalid-header", bad, len, sha, (uint64_t)len, false);
    free(bad);

    bad = copy_bytes(base, len);
    bad[3] = 2;
    write_malformed(fp, "bad_version", "invalid-header", bad, len, sha, (uint64_t)len, false);
    free(bad);

    bad = copy_bytes(base, len);
    put32le(bad + 8, 0);
    write_malformed(fp, "zero_tokens", "invalid-header", bad, len, sha, (uint64_t)len, false);
    free(bad);

    bad = copy_bytes(base, len);
    bad[4] = 3;
    write_malformed(fp, "bad_quant", "invalid-header", bad, len, sha, (uint64_t)len, false);
    free(bad);

    bad = copy_bytes(base, len);
    const uint32_t declared_text_len = (uint32_t)(sizeof(text) - 1 + 100);
    put32le(bad + DS4_KVSTORE_FIXED_HEADER, declared_text_len);
    write_malformed(fp, "declared_text_truncated", "truncated-text", bad, len, sha,
                    (uint64_t)len + 100, false);
    free(bad);

    bad = copy_bytes(base, len);
    put64le(bad + 40, sizeof(payload) + 99ull);
    write_malformed(fp, "declared_payload_truncated", "truncated-payload", bad, len, sha,
                    (uint64_t)len + 99, false);
    free(bad);

    write_malformed(fp, "declared_trailer_truncated", "truncated-trailer",
                    base, len - 1, sha, (uint64_t)len, false);
    fputs("\n  ]\n", fp);
    free(base);
}

static int kvc_file_dump_json(FILE *fp) {
    fputs("{\n", fp);
    fputs("  \"schema\": \"ds4.kvc_file_oracle.v1\",\n", fp);
    fputs("  \"source\": \"current-c-kvstore-file-no-model\",\n", fp);
    fputs("  \"model\": \"no model is loaded for this oracle\",\n", fp);
    fprintf(fp,
            "  \"constants\": {\"fixed_header\":%u,\"ext_flags\":{"
            "\"tool_map\":%u,\"responses_visible\":%u,"
            "\"thinking_visible\":%u}},\n",
            DS4_KVSTORE_FIXED_HEADER,
            DS4_KVSTORE_EXT_TOOL_MAP,
            DS4_KVSTORE_EXT_RESPONSES_VISIBLE,
            DS4_KVSTORE_EXT_THINKING_VISIBLE);
    write_cases(fp);
    write_malformed_cases(fp);
    fputs("}\n", fp);
    return ferror(fp) ? 1 : 0;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-kvc-file-dump [OUTPUT]\n"
            "\n"
            "Emit deterministic no-model JSON for generic KVC file layout.\n"
            "\n"
            "Arguments:\n"
            "  OUTPUT   Optional output file. Defaults to stdout.\n"
            "  -h, --help\n"
            "           Show this help\n");
}

int main(int argc, char **argv) {
    setlocale(LC_NUMERIC, "C");

    const char *output_path = NULL;
    if (argc > 2) {
        usage(stderr);
        return 2;
    }
    if (argc == 2) {
        if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
            usage(stdout);
            return 0;
        }
        output_path = argv[1];
    }

    FILE *fp = stdout;
    if (output_path) {
        fp = fopen(output_path, "wb");
        if (!fp) {
            fprintf(stderr, "ds4-kvc-file-dump: failed to open %s: %s\n",
                    output_path, strerror(errno));
            return 1;
        }
    }

    int rc = kvc_file_dump_json(fp);
    if (output_path && fclose(fp) != 0) {
        fprintf(stderr, "ds4-kvc-file-dump: failed to close %s: %s\n",
                output_path, strerror(errno));
        return 1;
    }
    return rc;
}
