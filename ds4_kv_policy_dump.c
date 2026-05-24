#include "ds4_kvstore.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *name;
    ds4_kvstore_options opt;
    bool enabled;
    uint64_t budget_bytes;
    int continued_last_store_tokens;
} kv_policy_config;

typedef struct {
    const char *name;
    const char *text;
    uint8_t quant_bits;
    uint8_t reason;
    uint8_t ext_flags;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    char sha[41];
    char *path;
} kv_fake_entry;

static char *find_prefix_tmp_dir;
static kv_fake_entry *find_prefix_tmp_entries;
static size_t find_prefix_tmp_len;
static bool find_prefix_cleanup_registered;

static void die(const char *msg) {
    fprintf(stderr, "ds4-kv-policy-dump: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg, const char *path) {
    fprintf(stderr, "ds4-kv-policy-dump: %s %s: %s\n",
            msg, path, strerror(errno));
    exit(1);
}

static void cleanup_find_prefix_tmp(void) {
    if (find_prefix_tmp_entries) {
        for (size_t i = 0; i < find_prefix_tmp_len; i++) {
            if (find_prefix_tmp_entries[i].path) {
                unlink(find_prefix_tmp_entries[i].path);
                free(find_prefix_tmp_entries[i].path);
                find_prefix_tmp_entries[i].path = NULL;
            }
        }
    }
    if (find_prefix_tmp_dir) {
        rmdir(find_prefix_tmp_dir);
        free(find_prefix_tmp_dir);
        find_prefix_tmp_dir = NULL;
    }
    find_prefix_tmp_entries = NULL;
    find_prefix_tmp_len = 0;
}

static char *kv_policy_tmp_template(void) {
    const char *base = getenv("TMPDIR");
    if (!base || !base[0]) base = "/tmp";
    size_t n = strlen(base);
    bool slash = n > 0 && base[n - 1] == '/';
    const char suffix[] = "ds4-kv-policy-XXXXXX";
    char *tmpl = malloc(n + (slash ? 0 : 1) + sizeof(suffix));
    if (!tmpl) die("out of memory");
    sprintf(tmpl, "%s%s%s", base, slash ? "" : "/", suffix);
    return tmpl;
}

static void json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '\\': fputs("\\\\", fp); break;
        case '"': fputs("\\\"", fp); break;
        case '\b': fputs("\\b", fp); break;
        case '\f': fputs("\\f", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
            if (*p < 0x20) fprintf(fp, "\\u%04x", *p);
            else fputc(*p, fp);
            break;
        }
    }
    fputc('"', fp);
}

static void write_hex(FILE *fp, const void *ptr, size_t len) {
    static const char h[] = "0123456789abcdef";
    const uint8_t *p = ptr;
    for (size_t i = 0; i < len; i++) {
        fputc(h[p[i] >> 4], fp);
        fputc(h[p[i] & 15], fp);
    }
}

static void json_hex_string(FILE *fp, const void *ptr, size_t len) {
    fputc('"', fp);
    write_hex(fp, ptr, len);
    fputc('"', fp);
}

static void write_options(FILE *fp, ds4_kvstore_options opt) {
    fprintf(fp,
            "{\"min_tokens\":%d,\"cold_max_tokens\":%d,"
            "\"continued_interval_tokens\":%d,\"boundary_trim_tokens\":%d,"
            "\"boundary_align_tokens\":%d}",
            opt.min_tokens,
            opt.cold_max_tokens,
            opt.continued_interval_tokens,
            opt.boundary_trim_tokens,
            opt.boundary_align_tokens);
}

static ds4_kvstore make_kvstore(kv_policy_config cfg) {
    ds4_kvstore kc;
    memset(&kc, 0, sizeof(kc));
    kc.enabled = cfg.enabled;
    kc.budget_bytes = cfg.budget_bytes;
    kc.opt = cfg.opt;
    kc.continued_last_store_tokens = cfg.continued_last_store_tokens;
    return kc;
}

static bool decode_header_bytes(const uint8_t header[DS4_KVSTORE_FIXED_HEADER],
                                uint32_t text_bytes,
                                ds4_kvstore_entry *entry_out,
                                uint32_t *text_out) {
    FILE *fp = tmpfile();
    if (!fp) die("tmpfile failed");
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, text_bytes);
    if (fwrite(header, 1, DS4_KVSTORE_FIXED_HEADER, fp) !=
            DS4_KVSTORE_FIXED_HEADER ||
        fwrite(tb, 1, sizeof(tb), fp) != sizeof(tb) ||
        fflush(fp) != 0 ||
        fseek(fp, 0, SEEK_SET) != 0)
    {
        fclose(fp);
        die("failed to prepare header decode file");
    }
    memset(entry_out, 0, sizeof(*entry_out));
    bool ok = ds4_kvstore_read_header(fp, entry_out, text_out);
    fclose(fp);
    return ok;
}

static void write_decoded_entry(FILE *fp, const ds4_kvstore_entry *e,
                                uint32_t text_bytes) {
    fprintf(fp,
            "{\"quant_bits\":%u,\"reason\":%u,\"ext_flags\":%u,"
            "\"tokens\":%u,\"hits\":%u,\"ctx_size\":%u,"
            "\"created_at\":%" PRIu64 ",\"last_used\":%" PRIu64 ","
            "\"payload_bytes\":%" PRIu64 ",\"text_bytes\":%u}",
            e->quant_bits,
            e->reason,
            e->ext_flags,
            e->tokens,
            e->hits,
            e->ctx_size,
            e->created_at,
            e->last_used,
            e->payload_bytes,
            text_bytes);
}

static void write_header_case(FILE *fp, const char *name,
                              uint8_t quant_bits, uint8_t reason,
                              uint8_t ext_flags, uint32_t tokens,
                              uint32_t hits, uint32_t ctx_size,
                              uint64_t created_at, uint64_t last_used,
                              uint64_t payload_bytes, uint32_t text_bytes,
                              bool first) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, quant_bits, reason, ext_flags, tokens, hits,
                            ctx_size, created_at, last_used, payload_bytes);
    ds4_kvstore_entry decoded;
    uint32_t decoded_text = 0;
    bool read_ok = decode_header_bytes(h, text_bytes, &decoded, &decoded_text);
    fprintf(fp, "%s    {\"name\":", first ? "" : ",\n");
    json_string(fp, name);
    fputs(",\"input\":{", fp);
    fprintf(fp,
            "\"quant_bits\":%u,\"reason\":%u,\"ext_flags\":%u,"
            "\"tokens\":%u,\"hits\":%u,\"ctx_size\":%u,"
            "\"created_at\":%" PRIu64 ",\"last_used\":%" PRIu64 ","
            "\"payload_bytes\":%" PRIu64 ",\"text_bytes\":%u},",
            quant_bits,
            reason,
            ext_flags,
            tokens,
            hits,
            ctx_size,
            created_at,
            last_used,
            payload_bytes,
            text_bytes);
    fputs("\"header_hex\":", fp);
    json_hex_string(fp, h, sizeof(h));
    fputs(",\"text_len_hex\":", fp);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, text_bytes);
    json_hex_string(fp, tb, sizeof(tb));
    fprintf(fp, ",\"read_ok\":%s", read_ok ? "true" : "false");
    if (read_ok) {
        fputs(",\"decoded\":", fp);
        write_decoded_entry(fp, &decoded, decoded_text);
    }
    fputc('}', fp);
}

static void write_invalid_header_case(FILE *fp, const char *name,
                                      int byte_index, uint8_t byte_value,
                                      uint8_t quant_bits, uint32_t tokens,
                                      bool first) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, quant_bits, DS4_KVSTORE_REASON_COLD, 0,
                            tokens, 0, 4096, 100, 101, 256);
    h[byte_index] = byte_value;
    ds4_kvstore_entry decoded;
    uint32_t decoded_text = 0;
    bool read_ok = decode_header_bytes(h, 12, &decoded, &decoded_text);
    fprintf(fp, "%s    {\"name\":", first ? "" : ",\n");
    json_string(fp, name);
    fputs(",\"header_hex\":", fp);
    json_hex_string(fp, h, sizeof(h));
    fprintf(fp, ",\"read_ok\":%s}", read_ok ? "true" : "false");
}

static void write_reason_cases(FILE *fp) {
    const char *names[] = {
        "cold", "continued", "evict", "shutdown",
        "agent-system", "agent-session", "other"
    };
    fputs("  \"reason_codes\": [\n", fp);
    fprintf(fp, "    {\"input\":null,\"code\":%u}", ds4_kvstore_reason_code(NULL));
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        fputs(",\n    {\"input\":", fp);
        json_string(fp, names[i]);
        fprintf(fp, ",\"code\":%u}", ds4_kvstore_reason_code(names[i]));
    }
    fputs("\n  ],\n", fp);
}

static void write_key_kind_cases(FILE *fp) {
    const uint8_t flags[] = {
        0,
        DS4_KVSTORE_EXT_TOOL_MAP,
        DS4_KVSTORE_EXT_RESPONSES_VISIBLE,
        DS4_KVSTORE_EXT_THINKING_VISIBLE,
        DS4_KVSTORE_EXT_TOOL_MAP | DS4_KVSTORE_EXT_RESPONSES_VISIBLE,
        DS4_KVSTORE_EXT_RESPONSES_VISIBLE | DS4_KVSTORE_EXT_THINKING_VISIBLE,
    };
    fputs("  \"key_kind_cases\": [\n", fp);
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++) {
        fprintf(fp, "%s    {\"ext_flags\":%u,\"key_kind\":",
                i ? ",\n" : "", flags[i]);
        json_string(fp, ds4_kvstore_key_kind(flags[i]));
        fputc('}', fp);
    }
    fputs("\n  ],\n", fp);
}

static void write_little_endian_cases(FILE *fp) {
    const uint32_t values[] = {0u, 1u, 0x12345678u, 0xffffffffu};
    fputs("  \"little_endian_cases\": [\n", fp);
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        uint8_t bytes[4];
        ds4_kvstore_le_put32(bytes, values[i]);
        uint32_t roundtrip = ds4_kvstore_le_get32(bytes);
        fprintf(fp, "%s    {\"value\":%u,\"hex\":",
                i ? ",\n" : "", values[i]);
        json_hex_string(fp, bytes, sizeof(bytes));
        fprintf(fp, ",\"roundtrip\":%u}", roundtrip);
    }
    fputs("\n  ],\n", fp);
}

static void write_sha_cases(FILE *fp) {
    const char *texts[] = {
        "",
        "cache prefix",
        "Alpha beta gamma",
        "<|tool|>{\"id\":\"abc\"}",
    };
    fputs("  \"sha_cases\": [\n", fp);
    for (size_t i = 0; i < sizeof(texts) / sizeof(texts[0]); i++) {
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(texts[i], strlen(texts[i]), sha);
        fprintf(fp, "%s    {\"name\":\"case%zu\",\"text_hex\":",
                i ? ",\n" : "", i);
        json_hex_string(fp, texts[i], strlen(texts[i]));
        fputs(",\"sha1\":", fp);
        json_string(fp, sha);
        fputc('}', fp);
    }
    fputs("\n  ],\n", fp);
}

static void write_name_cases(FILE *fp) {
    const char *valid = "ABCDEF0123456789ABCDEF0123456789ABCDEF01.kv";
    const char *invalid_short = "abc.kv";
    const char *invalid_suffix = "abcdef0123456789abcdef0123456789abcdef01.bin";
    const char *inputs[] = {valid, invalid_short, invalid_suffix};
    fputs("  \"name_cases\": [\n", fp);
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        char sha[41] = {0};
        bool ok = ds4_kvstore_sha_hex_name(inputs[i], sha);
        fprintf(fp, "%s    {\"input\":", i ? ",\n" : "");
        json_string(fp, inputs[i]);
        fprintf(fp, ",\"valid\":%s", ok ? "true" : "false");
        if (ok) {
            fputs(",\"sha\":", fp);
            json_string(fp, sha);
        }
        fputc('}', fp);
    }
    fputs("\n  ],\n", fp);
}

static void write_path_cases(FILE *fp) {
    char sha[41];
    ds4_kvstore_sha1_bytes_hex("cache prefix", strlen("cache prefix"), sha);
    char *joined_a = ds4_kvstore_path_join("cache", "file.kv");
    char *joined_b = ds4_kvstore_path_join("cache/", "file.kv");
    ds4_kvstore kc = {0};
    kc.dir = "cache";
    char *path = ds4_kvstore_path_for_sha(&kc, sha);
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    fputs("  \"path_cases\": [\n", fp);
    fputs("    {\"dir\":\"cache\",\"name\":\"file.kv\",\"joined\":", fp);
    json_string(fp, joined_a);
    fputs("},\n    {\"dir\":\"cache/\",\"name\":\"file.kv\",\"joined\":", fp);
    json_string(fp, joined_b);
    fputs("},\n    {\"dir\":\"cache\",\"sha\":", fp);
    json_string(fp, sha);
    fputs(",\"basename\":", fp);
    json_string(fp, base);
    fputs("}\n  ],\n", fp);
    free(joined_a);
    free(joined_b);
    free(path);
}

static void write_header_cases(FILE *fp) {
    fputs("  \"header_cases\": [\n", fp);
    write_header_case(fp, "cold_token_text",
                      2, DS4_KVSTORE_REASON_COLD, 0, 550, 1, 32768,
                      1779417499ull, 1779417514ull, 31526948ull, 2520, true);
    write_header_case(fp, "continued_tool_map",
                      4, DS4_KVSTORE_REASON_CONTINUED,
                      DS4_KVSTORE_EXT_TOOL_MAP, 2048, 3, 65536,
                      1000, 1200, 4096, 64, false);
    write_header_case(fp, "unknown_reason_normalized",
                      2, 99, DS4_KVSTORE_EXT_RESPONSES_VISIBLE,
                      9, 0, 4096, 1, 2, 3, 4, false);
    fputs("\n  ],\n", fp);

    fputs("  \"invalid_header_cases\": [\n", fp);
    write_invalid_header_case(fp, "bad_magic", 0, 'X', 2, 16, true);
    write_invalid_header_case(fp, "bad_version", 3, 2, 2, 16, false);
    write_invalid_header_case(fp, "zero_tokens", 8, 0, 2, 0, false);
    write_invalid_header_case(fp, "bad_quant", 4, 3, 3, 16, false);
    fputs("\n  ],\n", fp);
}

static kv_policy_config config_default(const char *name) {
    kv_policy_config cfg = {
        .name = name,
        .opt = ds4_kvstore_default_options(),
        .enabled = true,
        .budget_bytes = 0,
        .continued_last_store_tokens = 0,
    };
    return cfg;
}

static void write_store_len_cases(FILE *fp) {
    kv_policy_config def = config_default("default");
    kv_policy_config align_zero = config_default("align_zero");
    align_zero.opt.boundary_align_tokens = 0;
    struct {
        const char *name;
        kv_policy_config cfg;
        int tokens;
    } cases[] = {
        {"below_min", def, 500},
        {"at_min_plus_trim", def, 544},
        {"aligned_after_trim", def, 4096},
        {"larger_aligned_after_trim", def, 5000},
        {"align_zero_uses_trimmed_stable", align_zero, 1000},
    };
    fputs("    \"store_len\": [\n", fp);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ds4_kvstore kc = make_kvstore(cases[i].cfg);
        int got = ds4_kvstore_store_len(&kc, cases[i].tokens);
        fprintf(fp, "%s      {\"name\":", i ? ",\n" : "");
        json_string(fp, cases[i].name);
        fputs(",\"options\":", fp);
        write_options(fp, cases[i].cfg.opt);
        fprintf(fp, ",\"tokens\":%d,\"store_len\":%d}", cases[i].tokens, got);
    }
    fputs("\n    ],\n", fp);
}

static void write_chat_anchor_cases(FILE *fp) {
    kv_policy_config def = config_default("chat");
    def.opt.min_tokens = 2;
    kv_policy_config strict = config_default("chat_strict");
    strict.opt.min_tokens = 4;
    int toks_a[] = {10, 1, 11, 1, 20, 2, 30};
    int toks_b[] = {1, 10, 2};
    int toks_c[] = {10, 11, 12};
    int toks_d[] = {2, 1, 10};
    int toks_e[] = {1, 10, 1, 20, 1, 2};
    int toks_f[] = {10, 11, 1, 2};
    int toks_g[] = {1, 10, 1};
    struct {
        const char *name;
        kv_policy_config cfg;
        int *tokens;
        int len;
        int user_id;
        int assistant_id;
    } cases[] = {
        {"last_user_before_assistant", def, toks_a, 7, 1, 2},
        {"user_below_min", strict, toks_b, 3, 1, 2},
        {"missing_markers", def, toks_c, 3, 1, 2},
        {"assistant_first", def, toks_d, 3, 1, 2},
        {"multiple_users_before_assistant", def, toks_e, 6, 1, 2},
        {"exact_min_boundary", def, toks_f, 4, 1, 2},
        {"same_user_and_assistant_id", def, toks_g, 3, 1, 1},
    };
    fputs("    \"chat_anchor\": [\n", fp);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ds4_kvstore kc = make_kvstore(cases[i].cfg);
        ds4_tokens tokens = {
            .v = cases[i].tokens,
            .len = cases[i].len,
            .cap = cases[i].len,
        };
        int got = ds4_kvstore_chat_anchor_pos(&kc, &tokens,
                                              cases[i].user_id,
                                              cases[i].assistant_id);
        fprintf(fp, "%s      {\"name\":", i ? ",\n" : "");
        json_string(fp, cases[i].name);
        fputs(",\"tokens\":[", fp);
        for (int j = 0; j < cases[i].len; j++) {
            fprintf(fp, "%s%d", j ? "," : "", cases[i].tokens[j]);
        }
        fprintf(fp,
                "],\"min_tokens\":%d,\"user_token_id\":%d,"
                "\"assistant_token_id\":%d,\"anchor_pos\":%d}",
                cases[i].cfg.opt.min_tokens,
                cases[i].user_id,
                cases[i].assistant_id,
                got);
    }
    fputs("\n    ],\n", fp);
}

static void write_continued_cases(FILE *fp) {
    kv_policy_config def = config_default("continued");
    kv_policy_config already = def;
    already.continued_last_store_tokens = 10240;
    kv_policy_config disabled = def;
    disabled.enabled = false;
    kv_policy_config align_zero = def;
    align_zero.opt.boundary_align_tokens = 0;
    kv_policy_config no_interval = def;
    no_interval.opt.continued_interval_tokens = 0;
    struct {
        const char *name;
        kv_policy_config cfg;
        int live_tokens;
    } cases[] = {
        {"below_min", def, 511},
        {"unaligned_interval", def, 10000},
        {"aligned_interval", def, 10240},
        {"already_stored", already, 10240},
        {"disabled_store", disabled, 10240},
        {"align_zero_interval", align_zero, 10000},
        {"no_interval", no_interval, 10000},
    };
    fputs("    \"continued_store_target\": [\n", fp);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ds4_kvstore kc = make_kvstore(cases[i].cfg);
        int got = ds4_kvstore_continued_store_target(&kc, cases[i].live_tokens);
        fprintf(fp, "%s      {\"name\":", i ? ",\n" : "");
        json_string(fp, cases[i].name);
        fprintf(fp,
                ",\"enabled\":%s,\"live_tokens\":%d,"
                "\"continued_last_store_tokens\":%d,\"options\":",
                cases[i].cfg.enabled ? "true" : "false",
                cases[i].live_tokens,
                cases[i].cfg.continued_last_store_tokens);
        write_options(fp, cases[i].cfg.opt);
        fprintf(fp, ",\"target\":%d}", got);
    }
    fputs("\n    ],\n", fp);
}

static void write_continued_transition_event(FILE *fp,
                                             const char *op,
                                             int tokens,
                                             bool has_tokens,
                                             int old_frontier,
                                             bool has_old_frontier,
                                             int restore_old,
                                             bool has_restore_old,
                                             int restore_suppressed,
                                             bool has_restore_suppressed,
                                             int frontier,
                                             int target_probe,
                                             int target) {
    fputs("{\"op\":", fp);
    json_string(fp, op);
    if (has_tokens) fprintf(fp, ",\"tokens\":%d", tokens);
    if (has_old_frontier) {
        fprintf(fp, ",\"old_frontier\":%d", old_frontier);
    }
    if (has_restore_old) {
        fprintf(fp, ",\"restore_old_frontier\":%d", restore_old);
    }
    if (has_restore_suppressed) {
        fprintf(fp, ",\"restore_suppressed_tokens\":%d", restore_suppressed);
    }
    fprintf(fp,
            ",\"frontier\":%d,\"target_probe\":%d,\"target\":%d}",
            frontier,
            target_probe,
            target);
}

static void write_continued_transition_case_begin(FILE *fp,
                                                  const char *name,
                                                  int initial_frontier,
                                                  bool first) {
    fprintf(fp, "%s      {\"name\":", first ? "" : ",\n");
    json_string(fp, name);
    fprintf(fp, ",\"initial_frontier\":%d,\"events\":[", initial_frontier);
}

static void write_continued_transition_cases(FILE *fp) {
    const int probe = 10240;
    fputs("    \"continued_frontier_transitions\": [\n", fp);

    {
        kv_policy_config cfg = config_default("note_store_grows");
        cfg.continued_last_store_tokens = 4096;
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(fp, "note_store_grows", 4096, true);
        ds4_kvstore_note_store(&kc, probe);
        write_continued_transition_event(
            fp, "note_store", probe, true, 4096, true,
            0, false, 0, false, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputs("]}", fp);
    }

    {
        kv_policy_config cfg = config_default("note_store_ignores_lower");
        cfg.continued_last_store_tokens = probe;
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(fp, "note_store_ignores_lower", probe, false);
        ds4_kvstore_note_store(&kc, 4096);
        write_continued_transition_event(
            fp, "note_store", 4096, true, probe, true,
            0, false, 0, false, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputs("]}", fp);
    }

    {
        kv_policy_config cfg = config_default("suppress_fresh_frontier");
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(fp, "suppress_fresh_frontier", 0, false);
        int old = ds4_kvstore_suppress_continued_store(&kc, probe);
        write_continued_transition_event(
            fp, "suppress", probe, true, old, true,
            0, false, 0, false, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputc(',', fp);
        ds4_kvstore_restore_suppressed_continued(&kc, old, probe);
        write_continued_transition_event(
            fp, "restore_suppressed", 0, false, 0, false,
            old, true, probe, true, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputs("]}", fp);
    }

    {
        kv_policy_config cfg = config_default("suppress_already_stored_skip");
        cfg.continued_last_store_tokens = probe;
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(fp, "suppress_already_stored_skip", probe, false);
        int old = ds4_kvstore_suppress_continued_store(&kc, probe);
        write_continued_transition_event(
            fp, "suppress", probe, true, old, true,
            0, false, 0, false, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputc(',', fp);
        ds4_kvstore_restore_suppressed_continued(&kc, old, probe);
        write_continued_transition_event(
            fp, "restore_suppressed", 0, false, 0, false,
            old, true, probe, true, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputs("]}", fp);
    }

    {
        kv_policy_config cfg = config_default("suppress_unaligned_skip");
        cfg.continued_last_store_tokens = probe;
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(fp, "suppress_unaligned_skip", probe, false);
        int old = ds4_kvstore_suppress_continued_store(&kc, 18432);
        write_continued_transition_event(
            fp, "suppress", 18432, true, old, true,
            0, false, 0, false, kc.continued_last_store_tokens, 18432,
            ds4_kvstore_continued_store_target(&kc, 18432));
        fputs("]}", fp);
    }

    {
        kv_policy_config cfg = config_default("restore_ignores_mismatch");
        cfg.continued_last_store_tokens = probe;
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(fp, "restore_ignores_mismatch", probe, false);
        ds4_kvstore_restore_suppressed_continued(&kc, 4096, 20480);
        write_continued_transition_event(
            fp, "restore_suppressed", 0, false, 0, false,
            4096, true, 20480, true, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputs("]}", fp);
    }

    {
        kv_policy_config cfg = config_default("reset_after_miss");
        cfg.continued_last_store_tokens = 20480;
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(fp, "reset_after_miss", 20480, false);
        kc.continued_last_store_tokens = 0;
        write_continued_transition_event(
            fp, "reset_after_miss", 0, false, 0, false,
            0, false, 0, false, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputs("]}", fp);
    }

    {
        kv_policy_config cfg = config_default("disk_restore_records_loaded_frontier");
        ds4_kvstore kc = make_kvstore(cfg);
        write_continued_transition_case_begin(
            fp, "disk_restore_records_loaded_frontier", 0, false);
        kc.continued_last_store_tokens = 552;
        write_continued_transition_event(
            fp, "record_disk_load", 552, true, 0, false,
            0, false, 0, false, kc.continued_last_store_tokens, probe,
            ds4_kvstore_continued_store_target(&kc, probe));
        fputs("]}", fp);
    }

    fputs("\n    ],\n", fp);
}

static void write_file_size_cases(FILE *fp) {
    kv_policy_config no_budget = config_default("no_budget");
    kv_policy_config under = config_default("under_budget");
    under.budget_bytes = 1024;
    kv_policy_config over = config_default("over_budget");
    over.budget_bytes = 385;
    struct {
        const char *name;
        kv_policy_config cfg;
        uint64_t text_bytes;
        uint64_t payload_bytes;
        uint64_t trailer_bytes;
    } cases[] = {
        {"no_budget", no_budget, 100, 200, 30},
        {"under_budget_with_slack", under, 100, 200, 30},
        {"over_budget_with_slack", over, 100, 200, 30},
        {"overflow_text", no_budget, UINT64_MAX, 1, 1},
    };
    fputs("    \"file_size_fits\": [\n", fp);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ds4_kvstore kc = make_kvstore(cases[i].cfg);
        uint64_t file_bytes = 0;
        uint64_t required_bytes = 0;
        bool ok = ds4_kvstore_file_size_fits(&kc,
                                             cases[i].text_bytes,
                                             cases[i].payload_bytes,
                                             cases[i].trailer_bytes,
                                             &file_bytes,
                                             &required_bytes);
        fprintf(fp, "%s      {\"name\":", i ? ",\n" : "");
        json_string(fp, cases[i].name);
        fprintf(fp,
                ",\"budget_bytes\":%" PRIu64 ",\"text_bytes\":%" PRIu64
                ",\"payload_bytes\":%" PRIu64 ",\"trailer_bytes\":%" PRIu64
                ",\"fits\":%s,\"file_bytes\":%" PRIu64
                ",\"required_bytes\":%" PRIu64 "}",
                cases[i].cfg.budget_bytes,
                cases[i].text_bytes,
                cases[i].payload_bytes,
                cases[i].trailer_bytes,
                ok ? "true" : "false",
                file_bytes,
                required_bytes);
    }
    fputs("\n    ],\n", fp);
}

static void write_prefix_cases(FILE *fp) {
    struct {
        const char *name;
        const char *text;
        const char *prefix;
    } cases[] = {
        {"matching_prefix", "abcdef", "abc"},
        {"empty_prefix", "abcdef", ""},
        {"mismatch", "abcdef", "abd"},
        {"prefix_longer_than_text", "abc", "abcd"},
    };
    fputs("    \"byte_prefix_match\": [\n", fp);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        bool got = ds4_kvstore_byte_prefix_match(cases[i].text,
                                                 strlen(cases[i].text),
                                                 cases[i].prefix,
                                                 strlen(cases[i].prefix));
        fprintf(fp, "%s      {\"name\":", i ? ",\n" : "");
        json_string(fp, cases[i].name);
        fputs(",\"text_hex\":", fp);
        json_hex_string(fp, cases[i].text, strlen(cases[i].text));
        fputs(",\"prefix_hex\":", fp);
        json_hex_string(fp, cases[i].prefix, strlen(cases[i].prefix));
        fprintf(fp, ",\"matches\":%s}", got ? "true" : "false");
    }
    fputs("\n    ],\n", fp);
}

static void write_eviction_cases(FILE *fp) {
    const uint64_t now = 1000000;
    ds4_kvstore_entry entries[] = {
        {.tokens = 1000, .hits = 3, .file_size = 1000, .last_used = now},
        {.tokens = 1000, .hits = 3, .file_size = 1000,
         .last_used = now - DS4_KVSTORE_HIT_HALF_LIFE_SECONDS},
        {.tokens = 1000, .hits = 3, .file_size = 1000,
         .last_used = now - 10 * DS4_KVSTORE_HIT_HALF_LIFE_SECONDS},
        {.tokens = 1000, .hits = 3, .file_size = 1000},
        {.tokens = 1000, .hits = 3, .file_size = 0, .last_used = now},
    };
    const char *names[] = {
        "fresh_hits", "one_half_life", "stale_hits_floor",
        "zero_timestamp", "zero_file_size"
    };
    memcpy(entries[0].sha, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 41);
    memcpy(entries[1].sha, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 41);
    memcpy(entries[2].sha, "cccccccccccccccccccccccccccccccccccccccc", 41);
    memcpy(entries[3].sha, "dddddddddddddddddddddddddddddddddddddddd", 41);
    memcpy(entries[4].sha, "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", 41);
    fputs("    \"eviction_score\": [\n", fp);
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); i++) {
        double score = ds4_kvstore_entry_eviction_score(&entries[i], NULL,
                                                        NULL, now);
        fprintf(fp,
                "%s      {\"name\":\"%s\",\"now\":%" PRIu64
                ",\"tokens\":%u,\"hits\":%u,\"file_size\":%" PRIu64
                ",\"created_at\":%" PRIu64 ",\"last_used\":%" PRIu64
                ",\"score\":%.17g}",
                i ? ",\n" : "",
                names[i],
                now,
                entries[i].tokens,
                entries[i].hits,
                entries[i].file_size,
                entries[i].created_at,
                entries[i].last_used,
                score);
    }
    double protected_score =
        ds4_kvstore_entry_eviction_score(&entries[0], NULL, entries[0].sha, now);
    fprintf(fp,
            ",\n      {\"name\":\"protected_sha\",\"now\":%" PRIu64
            ",\"sha\":\"%s\",\"protected_sha\":\"%s\",\"score\":%.17g}",
            now, entries[0].sha, entries[0].sha, protected_score);
    fputs("\n    ],\n", fp);
}

static void write_payload(FILE *fp, uint64_t bytes) {
    for (uint64_t i = 0; i < bytes; i++) {
        uint8_t b = (uint8_t)(0xa0u + (uint8_t)(i & 0xfu));
        if (fwrite(&b, 1, 1, fp) != 1) die("failed to write payload");
    }
}

static void create_fake_entry(const char *dir, kv_fake_entry *e) {
    ds4_kvstore_sha1_bytes_hex(e->text, strlen(e->text), e->sha);
    char name[44];
    memcpy(name, e->sha, 40);
    memcpy(name + 40, ".kv", 4);
    e->path = ds4_kvstore_path_join(dir, name);
    FILE *fp = fopen(e->path, "wb");
    if (!fp) die_errno("failed to create", e->path);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, e->quant_bits, e->reason, e->ext_flags,
                            e->tokens, e->hits, e->ctx_size,
                            e->created_at, e->last_used, e->payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)strlen(e->text));
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(e->text, 1, strlen(e->text), fp) == strlen(e->text);
    if (ok) write_payload(fp, e->payload_bytes);
    if (fclose(fp) != 0) ok = false;
    if (!ok) die_errno("failed to write", e->path);
}

static void write_find_result(FILE *fp, ds4_kvstore *kc, const char *name,
                              const char *prompt, int quant_bits, int ctx_size,
                              bool first) {
    int idx = ds4_kvstore_find_text_prefix(kc, prompt, quant_bits, ctx_size);
    fprintf(fp,
            "%s      {\"name\":",
            first ? "" : ",\n");
    json_string(fp, name);
    fputs(",\"prompt_hex\":", fp);
    json_hex_string(fp, prompt, strlen(prompt));
    fprintf(fp,
            ",\"quant_bits\":%d,\"ctx_size\":%d,"
            "\"reject_different_quant\":%s,\"found\":%s",
            quant_bits,
            ctx_size,
            kc->reject_different_quant ? "true" : "false",
            idx >= 0 ? "true" : "false");
    if (idx >= 0) {
        ds4_kvstore_entry *e = &kc->entry[idx];
        fputs(",\"selected_sha\":", fp);
        json_string(fp, e->sha);
        fprintf(fp,
                ",\"selected_tokens\":%u,\"selected_text_bytes\":%" PRIu64
                ",\"selected_quant_bits\":%u,\"selected_ctx_size\":%u",
                e->tokens,
                e->text_bytes,
                e->quant_bits,
                e->ctx_size);
    }
    fputc('}', fp);
}

static void write_find_prefix_cases(FILE *fp) {
    char *tmpl = kv_policy_tmp_template();
    char *dir = mkdtemp(tmpl);
    if (!dir) die_errno("failed to create temp dir", tmpl);

    static kv_fake_entry entries[] = {
        {
            .name = "medium_prefix",
            .text = "Alpha beta",
            .quant_bits = 2,
            .reason = DS4_KVSTORE_REASON_COLD,
            .tokens = 600,
            .hits = 2,
            .ctx_size = 32768,
            .created_at = 1000,
            .last_used = 1000,
            .payload_bytes = 8,
        },
        {
            .name = "longest_prefix",
            .text = "Alpha beta gamma",
            .quant_bits = 2,
            .reason = DS4_KVSTORE_REASON_CONTINUED,
            .tokens = 700,
            .hits = 1,
            .ctx_size = 32768,
            .created_at = 1001,
            .last_used = 1001,
            .payload_bytes = 8,
        },
        {
            .name = "ctx_too_large_for_default",
            .text = "Alpha beta gamma delta",
            .quant_bits = 4,
            .reason = DS4_KVSTORE_REASON_COLD,
            .tokens = 800,
            .hits = 0,
            .ctx_size = 65536,
            .created_at = 1002,
            .last_used = 1002,
            .payload_bytes = 8,
        },
        {
            .name = "below_min_tokens",
            .text = "Short",
            .quant_bits = 2,
            .reason = DS4_KVSTORE_REASON_COLD,
            .tokens = 100,
            .hits = 0,
            .ctx_size = 32768,
            .created_at = 1003,
            .last_used = 1003,
            .payload_bytes = 8,
        },
    };
    find_prefix_tmp_dir = tmpl;
    find_prefix_tmp_entries = entries;
    find_prefix_tmp_len = sizeof(entries) / sizeof(entries[0]);
    if (!find_prefix_cleanup_registered) {
        atexit(cleanup_find_prefix_tmp);
        find_prefix_cleanup_registered = true;
    }

    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); i++) {
        create_fake_entry(dir, &entries[i]);
    }

    kv_policy_config cfg = config_default("find_text_prefix");
    ds4_kvstore kc = make_kvstore(cfg);
    kc.dir = dir;
    kc.reject_different_quant = true;
    fputs("    \"find_text_prefix\": [\n", fp);
    write_find_result(fp, &kc, "longest_prefix",
                      "Alpha beta gamma suffix", 2, 32768, true);
    write_find_result(fp, &kc, "reject_quant",
                      "Alpha beta gamma suffix", 4, 32768, false);
    kc.reject_different_quant = false;
    write_find_result(fp, &kc, "allow_cross_quant_when_config_accepts",
                      "Alpha beta gamma suffix", 4, 32768, false);
    kc.reject_different_quant = true;
    write_find_result(fp, &kc, "reject_ctx_too_small",
                      "Alpha beta gamma suffix", 2, 1024, false);
    write_find_result(fp, &kc, "reject_below_min",
                      "Short suffix", 2, 32768, false);
    write_find_result(fp, &kc, "longest_with_large_context",
                      "Alpha beta gamma delta suffix", 4, 65536, false);
    fputs("\n    ]\n", fp);

    kc.dir = NULL;
    ds4_kvstore_clear(&kc);
    cleanup_find_prefix_tmp();
}

static void write_policy_cases(FILE *fp) {
    fputs("  \"policy_cases\": {\n", fp);
    write_store_len_cases(fp);
    write_chat_anchor_cases(fp);
    write_continued_cases(fp);
    write_continued_transition_cases(fp);
    write_file_size_cases(fp);
    write_prefix_cases(fp);
    write_eviction_cases(fp);
    write_find_prefix_cases(fp);
    fputs("  },\n", fp);
}

static void write_m05_fixture(FILE *fp) {
    fputs("  \"m0_5_header_fixture\": {\n", fp);
    fputs("    \"path\":\"ds4-parity/baselines/kv-artifacts/m0.5/logs/kv-header.tsv\",\n", fp);
    fputs("    \"expected_rows\": [\n", fp);
    fputs("      {\"file\":\"0ab2314538b11686a11e296b7f697651fbd17e60.kv\","
          "\"quant\":2,\"reason\":1,\"reason_name\":\"cold\",\"ext_flags\":0,"
          "\"tokens\":550,\"hits\":1,\"ctx\":32768,\"payload_bytes\":31526948,"
          "\"rendered_text_bytes\":2520,\"trailer_bytes\":0},\n", fp);
    fputs("      {\"file\":\"4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.kv\","
          "\"quant\":2,\"reason\":4,\"reason_name\":\"shutdown\","
          "\"ext_flags\":0,\"tokens\":563,\"hits\":0,\"ctx\":32768,"
          "\"payload_bytes\":31688280,\"rendered_text_bytes\":2632,"
          "\"trailer_bytes\":0},\n", fp);
    fputs("      {\"file\":\"a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv\","
          "\"quant\":2,\"reason\":4,\"reason_name\":\"shutdown\","
          "\"ext_flags\":0,\"tokens\":552,\"hits\":1,\"ctx\":32768,"
          "\"payload_bytes\":31580716,\"rendered_text_bytes\":2528,"
          "\"trailer_bytes\":0}\n", fp);
    fputs("    ]\n", fp);
    fputs("  }\n", fp);
}

static int kv_policy_dump_json(FILE *fp) {
    ds4_kvstore_options defaults = ds4_kvstore_default_options();
    fputs("{\n", fp);
    fputs("  \"schema\": \"ds4.kv_policy_oracle.v1\",\n", fp);
    fputs("  \"source\": \"current-c-kvstore-no-model\",\n", fp);
    fputs("  \"model\": \"no model is loaded for this oracle\",\n", fp);
    fprintf(fp,
            "  \"constants\": {\"fixed_header\":%u,\"default_mb\":%u,"
            "\"hit_half_life_seconds\":%llu,\"ext_flags\":{"
            "\"tool_map\":%u,\"responses_visible\":%u,"
            "\"thinking_visible\":%u}},\n",
            DS4_KVSTORE_FIXED_HEADER,
            DS4_KVSTORE_DEFAULT_MB,
            (unsigned long long)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS,
            DS4_KVSTORE_EXT_TOOL_MAP,
            DS4_KVSTORE_EXT_RESPONSES_VISIBLE,
            DS4_KVSTORE_EXT_THINKING_VISIBLE);
    fputs("  \"defaults\": ", fp);
    write_options(fp, defaults);
    fputs(",\n", fp);
    write_reason_cases(fp);
    write_key_kind_cases(fp);
    write_little_endian_cases(fp);
    write_sha_cases(fp);
    write_name_cases(fp);
    write_path_cases(fp);
    write_header_cases(fp);
    write_policy_cases(fp);
    write_m05_fixture(fp);
    fputs("}\n", fp);
    return ferror(fp) ? 1 : 0;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-kv-policy-dump [OUTPUT]\n"
            "\n"
            "Emit deterministic no-model JSON for current-C KV header and policy helpers.\n"
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
            fprintf(stderr, "ds4-kv-policy-dump: failed to open %s: %s\n",
                    output_path, strerror(errno));
            return 1;
        }
    }

    int rc = kv_policy_dump_json(fp);
    if (output_path && fclose(fp) != 0) {
        fprintf(stderr, "ds4-kv-policy-dump: failed to close %s: %s\n",
                output_path, strerror(errno));
        return 1;
    }
    return rc;
}
