#include "ds4.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RESTORE_TOP_K 20
#define RESTORE_HEADER_PREFIX 52

typedef struct {
    const char *model_path;
    const char *output_path;
    const char *payload_dir;
    const char *seed_prompt_path;
    const char *seed_assistant_path;
    const char *continuation_user_path;
    const char *model_sha256;
    ds4_backend backend;
} restore_options;

typedef struct {
    int id;
    float logit;
    float logprob;
    char bytes_hex[256];
} restore_score;

typedef struct {
    int selected_token;
    char selected_bytes_hex[256];
    int top_count;
    restore_score top[RESTORE_TOP_K];
} restore_state;

typedef struct {
    bool selected_match;
    bool top_order_match;
    float max_abs_logit_delta;
    float max_abs_logprob_delta;
} restore_comparison;

typedef struct {
    char *ptr;
    size_t len;
} text_buf;

static void die(const char *msg) {
    fprintf(stderr, "ds4-restore-dump: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg, const char *path) {
    fprintf(stderr, "ds4-restore-dump: %s %s: %s\n", msg, path, strerror(errno));
    exit(1);
}

static char *shell_quote(const char *s) {
    size_t n = 2;
    for (const char *p = s; *p; p++) n += *p == '\'' ? 4 : 1;
    char *out = malloc(n + 1);
    if (!out) die("out of memory");
    char *w = out;
    *w++ = '\'';
    for (const char *p = s; *p; p++) {
        if (*p == '\'') {
            memcpy(w, "'\\''", 4);
            w += 4;
        } else {
            *w++ = *p;
        }
    }
    *w++ = '\'';
    *w = '\0';
    return out;
}

static bool is_sha256_hex(const char *s) {
    if (!s || strlen(s) != 64) return false;
    for (const char *p = s; *p; p++) {
        if (!(*p >= '0' && *p <= '9') && !(*p >= 'a' && *p <= 'f')) return false;
    }
    return true;
}

static bool read_sha256_command(const char *cmd, char out[65]) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return false;
    char line[256];
    bool ok = fgets(line, sizeof(line), fp) != NULL;
    const int rc = pclose(fp);
    if (!ok || rc != 0) return false;
    line[strcspn(line, " \t\r\n")] = '\0';
    if (!is_sha256_hex(line)) return false;
    memcpy(out, line, 65);
    return true;
}

static bool file_sha256_hex(const char *path, char out[65]) {
    char *quoted = shell_quote(path);
    const size_t n = strlen(quoted) + 64;
    char *cmd = malloc(n);
    if (!cmd) die("out of memory");
    snprintf(cmd, n, "sha256sum %s 2>/dev/null", quoted);
    bool ok = read_sha256_command(cmd, out);
    if (!ok) {
        snprintf(cmd, n, "shasum -a 256 %s 2>/dev/null", quoted);
        ok = read_sha256_command(cmd, out);
    }
    free(cmd);
    free(quoted);
    return ok;
}

static void verify_model_sha256(const char *path, const char *expected) {
    char actual[65];
    if (!file_sha256_hex(path, actual)) {
        die("failed to verify model sha256 with sha256sum or shasum");
    }
    if (strcmp(actual, expected) != 0) {
        fprintf(stderr,
                "ds4-restore-dump: model sha256 mismatch: expected %s actual %s\n",
                expected,
                actual);
        exit(1);
    }
}

static text_buf read_text_file(const char *path, bool trim_trailing_newline) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("failed to open", path);
    if (fseek(fp, 0, SEEK_END) != 0) die_errno("failed to seek", path);
    long n = ftell(fp);
    if (n < 0) die_errno("failed to tell", path);
    if (fseek(fp, 0, SEEK_SET) != 0) die_errno("failed to rewind", path);
    char *buf = malloc((size_t)n + 1);
    if (!buf) die("out of memory");
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) die_errno("failed to read", path);
    if (fclose(fp) != 0) die_errno("failed to close", path);
    size_t len = (size_t)n;
    if (trim_trailing_newline) {
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) len--;
    }
    buf[len] = '\0';
    return (text_buf){.ptr = buf, .len = len};
}

static void bytes_to_hex(const void *ptr, size_t len, char *out, size_t cap) {
    static const char h[] = "0123456789abcdef";
    const uint8_t *bytes = ptr;
    if (cap < len * 2 + 1) die("hex output buffer too small");
    for (size_t i = 0; i < len; i++) {
        out[i * 2] = h[bytes[i] >> 4];
        out[i * 2 + 1] = h[bytes[i] & 0xf];
    }
    out[len * 2] = '\0';
}

static void json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', fp);
            fputc(*p, fp);
        } else if (*p == '\n') {
            fputs("\\n", fp);
        } else if (*p == '\r') {
            fputs("\\r", fp);
        } else if (*p == '\t') {
            fputs("\\t", fp);
        } else if (*p < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)*p);
        } else {
            fputc(*p, fp);
        }
    }
    fputc('"', fp);
}

static void json_f32(FILE *fp, float v) {
    if (isnan(v)) {
        fputs("\"nan\"", fp);
    } else if (isinf(v)) {
        fputs(signbit(v) ? "\"-inf\"" : "\"inf\"", fp);
    } else {
        fprintf(fp, "%.9g", v);
    }
}

static void token_hex_string(ds4_engine *engine, int token, char out[256]) {
    size_t len = 0;
    char *text = ds4_token_text(engine, token, &len);
    if (len * 2 + 1 > 256) die("token text is too large for fixture");
    bytes_to_hex(text, len, out, 256);
    free(text);
}

static void capture_state(ds4_engine *engine, ds4_session *session, restore_state *state) {
    memset(state, 0, sizeof(*state));
    state->selected_token = ds4_session_argmax(session);
    token_hex_string(engine, state->selected_token, state->selected_bytes_hex);
    ds4_token_score scores[RESTORE_TOP_K];
    state->top_count = ds4_session_top_logprobs(session, scores, RESTORE_TOP_K);
    if (state->top_count != RESTORE_TOP_K) die("unexpected top-logprob count");
    for (int i = 0; i < state->top_count; i++) {
        state->top[i].id = scores[i].id;
        state->top[i].logit = scores[i].logit;
        state->top[i].logprob = scores[i].logprob;
        token_hex_string(engine, scores[i].id, state->top[i].bytes_hex);
    }
}

static restore_comparison compare_states(const restore_state *ref, const restore_state *restored) {
    restore_comparison cmp = {
        .selected_match = ref->selected_token == restored->selected_token,
        .top_order_match = ref->top_count == restored->top_count,
    };
    const int n = ref->top_count < restored->top_count ? ref->top_count : restored->top_count;
    for (int i = 0; i < n; i++) {
        if (ref->top[i].id != restored->top[i].id) cmp.top_order_match = false;
        const float logit_delta = fabsf(ref->top[i].logit - restored->top[i].logit);
        const float logprob_delta = fabsf(ref->top[i].logprob - restored->top[i].logprob);
        if (logit_delta > cmp.max_abs_logit_delta) cmp.max_abs_logit_delta = logit_delta;
        if (logprob_delta > cmp.max_abs_logprob_delta) cmp.max_abs_logprob_delta = logprob_delta;
    }
    return cmp;
}

static void write_score(FILE *fp, const restore_score *score) {
    fprintf(fp, "{\"id\":%d,\"bytes_hex\":\"%s\",\"logit\":", score->id, score->bytes_hex);
    json_f32(fp, score->logit);
    fputs(",\"logprob\":", fp);
    json_f32(fp, score->logprob);
    fputc('}', fp);
}

static void write_state(FILE *fp, const restore_state *state) {
    fprintf(fp, "{\"selected_token\":%d,\"selected_bytes_hex\":\"%s\",\"top_logprobs\":[",
            state->selected_token,
            state->selected_bytes_hex);
    for (int i = 0; i < state->top_count; i++) {
        if (i) fputc(',', fp);
        write_score(fp, &state->top[i]);
    }
    fputs("]}", fp);
}

static void write_comparison(FILE *fp, const restore_comparison *cmp) {
    fprintf(fp,
            "{\"selected_match\":%s,\"top_order_match\":%s,\"max_abs_logit_delta\":",
            cmp->selected_match ? "true" : "false",
            cmp->top_order_match ? "true" : "false");
    json_f32(fp, cmp->max_abs_logit_delta);
    fputs(",\"max_abs_logprob_delta\":", fp);
    json_f32(fp, cmp->max_abs_logprob_delta);
    fputc('}', fp);
}

static void build_seed_prompt(ds4_engine *engine, const text_buf *seed, ds4_tokens *out) {
    ds4_chat_begin(engine, out);
    ds4_chat_append_message(engine, out, "user", seed->ptr);
    ds4_chat_append_assistant_prefix(engine, out, DS4_THINK_NONE);
}

static void build_continuation_prompt(ds4_engine *engine,
                                      const text_buf *seed,
                                      const text_buf *assistant,
                                      const text_buf *continuation,
                                      ds4_tokens *out) {
    ds4_chat_begin(engine, out);
    ds4_chat_append_message(engine, out, "user", seed->ptr);
    ds4_chat_append_assistant_prefix(engine, out, DS4_THINK_NONE);
    ds4_tokenize_text(engine, assistant->ptr, out);
    ds4_tokens_push(out, ds4_token_eos(engine));
    ds4_chat_append_message(engine, out, "user", continuation->ptr);
    ds4_chat_append_assistant_prefix(engine, out, DS4_THINK_NONE);
}

static void sync_prompt(ds4_session *session, const ds4_tokens *prompt, const char *case_id) {
    char err[256];
    if (ds4_session_sync(session, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-restore-dump: sync failed for %s: %s\n", case_id, err);
        exit(1);
    }
}

static void load_payload_file(ds4_session *session, const char *path, uint64_t bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("failed to open payload", path);
    char err[256];
    const int rc = ds4_session_load_payload(session, fp, bytes, err, sizeof(err));
    if (fclose(fp) != 0 && rc == 0) die_errno("failed to close payload", path);
    if (rc != 0) {
        fprintf(stderr, "ds4-restore-dump: payload restore failed for %s: %s\n", path, err);
        exit(1);
    }
}

static uint8_t *read_file_bytes(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("failed to open", path);
    if (fseek(fp, 0, SEEK_END) != 0) die_errno("failed to seek", path);
    long n = ftell(fp);
    if (n < 0) die_errno("failed to tell", path);
    if (fseek(fp, 0, SEEK_SET) != 0) die_errno("failed to rewind", path);
    uint8_t *buf = malloc((size_t)n);
    if (!buf && n > 0) die("out of memory");
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) die_errno("failed to read", path);
    if (fclose(fp) != 0) die_errno("failed to close", path);
    *out_len = (size_t)n;
    return buf;
}

static void write_disk_payload_case(FILE *json,
                                    ds4_engine *engine,
                                    const restore_options *opt,
                                    const char *case_id,
                                    const char *prompt_case,
                                    const ds4_tokens *prompt,
                                    bool *first_case) {
    ds4_session *source = NULL;
    ds4_session *restored = NULL;
    if (ds4_session_create(&source, engine, 32768) != 0 || !source) die("failed to create source session");
    if (ds4_session_create(&restored, engine, 32768) != 0 || !restored) die("failed to create restored session");
    sync_prompt(source, prompt, case_id);

    char payload_path[1024];
    snprintf(payload_path, sizeof(payload_path), "%s/%s.dsv4", opt->payload_dir, case_id);
    FILE *payload = fopen(payload_path, "wb");
    if (!payload) die_errno("failed to create payload", payload_path);
    char err[256];
    if (ds4_session_save_payload(source, payload, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-restore-dump: payload save failed for %s: %s\n", case_id, err);
        exit(1);
    }
    if (fclose(payload) != 0) die_errno("failed to close payload", payload_path);

    size_t payload_len = 0;
    uint8_t *payload_bytes = read_file_bytes(payload_path, &payload_len);
    char payload_sha[65];
    ds4_sha256_hex(payload_bytes, payload_len, payload_sha);
    char header_hex[RESTORE_HEADER_PREFIX * 2 + 1];
    bytes_to_hex(payload_bytes,
                 payload_len < RESTORE_HEADER_PREFIX ? payload_len : RESTORE_HEADER_PREFIX,
                 header_hex,
                 sizeof(header_hex));

    load_payload_file(restored, payload_path, (uint64_t)payload_len);
    restore_state reference;
    restore_state restored_state;
    capture_state(engine, source, &reference);
    capture_state(engine, restored, &restored_state);
    restore_comparison cmp = compare_states(&reference, &restored_state);

    if (!*first_case) fputs(",\n", json);
    *first_case = false;
    fputs("    {\"id\":", json);
    json_string(json, case_id);
    fputs(",\"kind\":\"disk-payload\",\"prompt_case\":", json);
    json_string(json, prompt_case);
    fprintf(json,
            ",\"ctx\":32768,\"prompt_tokens\":%d,\"payload_file\":",
            prompt->len);
    json_string(json, payload_path);
    fprintf(json,
            ",\"raw_payload_committed\":false,\"payload_bytes\":%zu,\"payload_sha256\":\"%s\",\"header_prefix_hex\":\"%s\",",
            payload_len,
            payload_sha,
            header_hex);
    fputs("\"reference\":", json);
    write_state(json, &reference);
    fputs(",\"restored\":", json);
    write_state(json, &restored_state);
    fputs(",\"comparison\":", json);
    write_comparison(json, &cmp);
    fputc('}', json);

    free(payload_bytes);
    ds4_session_free(source);
    ds4_session_free(restored);
}

static void write_memory_snapshot_case(FILE *json,
                                       ds4_engine *engine,
                                       const char *case_id,
                                       const char *prompt_case,
                                       const ds4_tokens *prompt,
                                       bool *first_case) {
    ds4_session *source = NULL;
    ds4_session *restored = NULL;
    if (ds4_session_create(&source, engine, 32768) != 0 || !source) die("failed to create source session");
    if (ds4_session_create(&restored, engine, 32768) != 0 || !restored) die("failed to create restored session");
    sync_prompt(source, prompt, case_id);

    ds4_session_snapshot snap = {0};
    char err[256];
    if (ds4_session_save_snapshot(source, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-restore-dump: snapshot save failed for %s: %s\n", case_id, err);
        exit(1);
    }
    if (ds4_session_load_snapshot(restored, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-restore-dump: snapshot restore failed for %s: %s\n", case_id, err);
        exit(1);
    }
    char snapshot_sha[65];
    ds4_sha256_hex(snap.ptr, (size_t)snap.len, snapshot_sha);
    char header_hex[RESTORE_HEADER_PREFIX * 2 + 1];
    bytes_to_hex(snap.ptr,
                 snap.len < RESTORE_HEADER_PREFIX ? (size_t)snap.len : RESTORE_HEADER_PREFIX,
                 header_hex,
                 sizeof(header_hex));

    restore_state reference;
    restore_state restored_state;
    capture_state(engine, source, &reference);
    capture_state(engine, restored, &restored_state);
    restore_comparison cmp = compare_states(&reference, &restored_state);

    if (!*first_case) fputs(",\n", json);
    *first_case = false;
    fputs("    {\"id\":", json);
    json_string(json, case_id);
    fputs(",\"kind\":\"memory-snapshot\",\"prompt_case\":", json);
    json_string(json, prompt_case);
    fprintf(json,
            ",\"ctx\":32768,\"prompt_tokens\":%d,\"raw_payload_committed\":false,\"snapshot_bytes\":%" PRIu64 ",\"snapshot_cap\":%" PRIu64 ",\"snapshot_sha256\":\"%s\",\"header_prefix_hex\":\"%s\",",
            prompt->len,
            snap.len,
            snap.cap,
            snapshot_sha,
            header_hex);
    fputs("\"reference\":", json);
    write_state(json, &reference);
    fputs(",\"restored\":", json);
    write_state(json, &restored_state);
    fputs(",\"comparison\":", json);
    write_comparison(json, &cmp);
    fputc('}', json);

    ds4_session_snapshot_free(&snap);
    ds4_session_free(source);
    ds4_session_free(restored);
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    die("unknown backend");
    return DS4_BACKEND_CPU;
}

static void usage(FILE *fp) {
    fputs("usage: ds4-restore-dump -m MODEL -o OUTPUT.json --payload-dir DIR --seed-prompt FILE --seed-assistant FILE --continuation-user FILE --model-sha256 HEX [--backend cuda|metal|cpu]\n", fp);
}

static restore_options parse_args(int argc, char **argv) {
    restore_options opt = {
        .backend = DS4_BACKEND_METAL,
        .model_sha256 = "",
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            if (++i >= argc) die("missing model path");
            opt.model_path = argv[i];
        } else if (!strcmp(arg, "-o") || !strcmp(arg, "--output")) {
            if (++i >= argc) die("missing output path");
            opt.output_path = argv[i];
        } else if (!strcmp(arg, "--payload-dir")) {
            if (++i >= argc) die("missing payload dir");
            opt.payload_dir = argv[i];
        } else if (!strcmp(arg, "--seed-prompt")) {
            if (++i >= argc) die("missing seed prompt");
            opt.seed_prompt_path = argv[i];
        } else if (!strcmp(arg, "--seed-assistant")) {
            if (++i >= argc) die("missing seed assistant");
            opt.seed_assistant_path = argv[i];
        } else if (!strcmp(arg, "--continuation-user")) {
            if (++i >= argc) die("missing continuation user");
            opt.continuation_user_path = argv[i];
        } else if (!strcmp(arg, "--backend")) {
            if (++i >= argc) die("missing backend");
            opt.backend = parse_backend(argv[i]);
        } else if (!strcmp(arg, "--model-sha256")) {
            if (++i >= argc) die("missing model sha256");
            opt.model_sha256 = argv[i];
        } else {
            usage(stderr);
            exit(2);
        }
    }
    if (!opt.model_path || !opt.output_path || !opt.payload_dir ||
        !opt.seed_prompt_path || !opt.seed_assistant_path ||
        !opt.continuation_user_path || !opt.model_sha256[0])
    {
        usage(stderr);
        exit(2);
    }
    if (!is_sha256_hex(opt.model_sha256)) die("model sha256 must be 64 lowercase hex characters");
    return opt;
}

int main(int argc, char **argv) {
    restore_options opt = parse_args(argc, argv);
    verify_model_sha256(opt.model_path, opt.model_sha256);

    text_buf seed = read_text_file(opt.seed_prompt_path, true);
    text_buf assistant = read_text_file(opt.seed_assistant_path, true);
    text_buf continuation = read_text_file(opt.continuation_user_path, true);

    ds4_engine *engine = NULL;
    ds4_engine_options eopt = {
        .model_path = opt.model_path,
        .backend = opt.backend,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
    };
    if (ds4_engine_open(&engine, &eopt) != 0 || !engine) die("failed to open model");

    ds4_tokens seed_prompt = {0};
    ds4_tokens continuation_prompt = {0};
    build_seed_prompt(engine, &seed, &seed_prompt);
    build_continuation_prompt(engine, &seed, &assistant, &continuation, &continuation_prompt);

    FILE *json = fopen(opt.output_path, "wb");
    if (!json) die_errno("failed to open output", opt.output_path);
    char seed_sha[65];
    char assistant_sha[65];
    char continuation_sha[65];
    ds4_sha256_hex(seed.ptr, seed.len, seed_sha);
    ds4_sha256_hex(assistant.ptr, assistant.len, assistant_sha);
    ds4_sha256_hex(continuation.ptr, continuation.len, continuation_sha);

    fputs("{\n", json);
    fputs("  \"schema\": \"ds4.restore_oracle.v1\",\n", json);
    fputs("  \"source\": \"current-c-b300-restore\",\n", json);
    fputs("  \"model\": \"deepseek-v4-flash\",\n", json);
    fputs("  \"model_path\": ", json);
    json_string(json, opt.model_path);
    fputs(",\n  \"model_sha256\": ", json);
    json_string(json, opt.model_sha256);
    fputs(",\n  \"backend\": ", json);
    json_string(json, ds4_backend_name(opt.backend));
    fputs(",\n  \"top_k\": 20,\n", json);
    fputs("  \"score_abs_tolerance\": 1e-5,\n", json);
    fputs("  \"fixtures\": {\n", json);
    fputs("    \"seed_prompt\": {\"path\": ", json);
    json_string(json, opt.seed_prompt_path);
    fprintf(json, ", \"sha256\": \"%s\", \"trim_trailing_newline\": true},\n", seed_sha);
    fputs("    \"seed_assistant\": {\"path\": ", json);
    json_string(json, opt.seed_assistant_path);
    fprintf(json, ", \"sha256\": \"%s\", \"trim_trailing_newline\": true},\n", assistant_sha);
    fputs("    \"continuation_user\": {\"path\": ", json);
    json_string(json, opt.continuation_user_path);
    fprintf(json, ", \"sha256\": \"%s\", \"trim_trailing_newline\": true}\n", continuation_sha);
    fputs("  },\n", json);
    fputs("  \"cases\": [\n", json);

    bool first_case = true;
    write_disk_payload_case(json, engine, &opt, "disk_seed_payload", "seed", &seed_prompt, &first_case);
    write_memory_snapshot_case(json, engine, "snapshot_seed", "seed", &seed_prompt, &first_case);
    write_disk_payload_case(json, engine, &opt, "disk_continuation_payload", "continuation", &continuation_prompt, &first_case);
    write_memory_snapshot_case(json, engine, "snapshot_continuation", "continuation", &continuation_prompt, &first_case);

    fputs("\n  ]\n}\n", json);
    if (fclose(json) != 0) die_errno("failed to close output", opt.output_path);

    ds4_tokens_free(&seed_prompt);
    ds4_tokens_free(&continuation_prompt);
    ds4_engine_close(engine);
    free(seed.ptr);
    free(assistant.ptr);
    free(continuation.ptr);
    return 0;
}
