#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "ds4_server.c"

#include <errno.h>

typedef enum {
    POLICY_SURFACE_CLI,
    POLICY_SURFACE_SERVER,
    POLICY_SURFACE_AGENT,
} policy_surface;

typedef struct {
    const char *text;
    bool eos;
} policy_piece;

typedef struct {
    const char *name;
    const char *source;
    policy_surface surface;
    api_style api;
    req_kind kind;
    bool stream;
    bool has_tools;
    int max_tokens;
    const char * const *stops;
    int stop_count;
    const policy_piece *pieces;
    int piece_count;
} policy_case;

typedef struct {
    buf raw_text;
    buf visible_text;
    buf reasoning_text;
    buf streamed_text;
    buf stream_steps;
    char *finish;
    bool session_invalidated;
    bool transcript_eos_appended;
    bool saw_tool_start;
    bool saw_tool_end;
    int completion_tokens;
    int stop_pos;
    int stop_len;
    int tool_call_count;
} policy_result;

static const policy_piece CLI_EOS_PIECES[] = {
    {.text = "cli hello"},
    {.eos = true},
};

static const policy_piece CLI_LENGTH_PIECES[] = {
    {.text = "a"},
    {.text = "b"},
    {.text = "c"},
};

static const policy_piece SERVER_EOS_PIECES[] = {
    {.text = "server hello"},
    {.eos = true},
};

static const policy_piece SERVER_LENGTH_PIECES[] = {
    {.text = "one"},
    {.text = " two"},
    {.text = " three"},
};

static const policy_piece SERVER_STOP_PIECES[] = {
    {.text = "answer ST"},
    {.text = "OP hidden"},
};

static const policy_piece SERVER_STOP_HOLD_PIECES[] = {
    {.text = "hello </"},
    {.eos = true},
};

static const policy_piece SERVER_STREAM_STOP_HIT_PIECES[] = {
    {.text = "pre ST"},
    {.text = "OP after"},
};

static const policy_piece SERVER_UTF8_HOLD_PIECES[] = {
    {.text = "\xe2\x82"},
    {.text = "\xac ok"},
    {.eos = true},
};

static const policy_piece SERVER_STOP_MID_UTF8_PIECES[] = {
    {.text = "\xe2ST"},
    {.text = "OP tail"},
};

static const policy_piece SERVER_TOOL_PIECES[] = {
    {.text = "I will call.\n\n"},
    {.text =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">echo hi" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END},
};

static const policy_piece AGENT_EOS_PIECES[] = {
    {.text = "agent hello"},
    {.eos = true},
};

static const policy_piece AGENT_LENGTH_PIECES[] = {
    {.text = "x"},
    {.text = "y"},
    {.text = "z"},
};

static const char * const STOP_WORD[] = {"STOP"};
static const char * const STOP_END[] = {"</END>"};

static const policy_case POLICY_CASES[] = {
    {
        .name = "cli_eos_stop",
        .source = "ds4_cli.c:run_sampled_generation EOS",
        .surface = POLICY_SURFACE_CLI,
        .api = API_OPENAI,
        .kind = REQ_COMPLETION,
        .max_tokens = 4,
        .pieces = CLI_EOS_PIECES,
        .piece_count = (int)(sizeof(CLI_EOS_PIECES) / sizeof(CLI_EOS_PIECES[0])),
    },
    {
        .name = "cli_max_tokens_length",
        .source = "ds4_cli.c:run_sampled_generation max_tokens",
        .surface = POLICY_SURFACE_CLI,
        .api = API_OPENAI,
        .kind = REQ_COMPLETION,
        .max_tokens = 2,
        .pieces = CLI_LENGTH_PIECES,
        .piece_count = (int)(sizeof(CLI_LENGTH_PIECES) / sizeof(CLI_LENGTH_PIECES[0])),
    },
    {
        .name = "server_openai_eos_stop",
        .source = "ds4_server.c:generate_job EOS",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .max_tokens = 4,
        .pieces = SERVER_EOS_PIECES,
        .piece_count = (int)(sizeof(SERVER_EOS_PIECES) / sizeof(SERVER_EOS_PIECES[0])),
    },
    {
        .name = "server_openai_max_tokens_length",
        .source = "ds4_server.c:generate_job max_tokens",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .max_tokens = 2,
        .pieces = SERVER_LENGTH_PIECES,
        .piece_count = (int)(sizeof(SERVER_LENGTH_PIECES) / sizeof(SERVER_LENGTH_PIECES[0])),
    },
    {
        .name = "server_openai_user_stop_sequence",
        .source = "ds4_server.c:generate_job stop_list_find_from",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .max_tokens = 4,
        .stops = STOP_WORD,
        .stop_count = (int)(sizeof(STOP_WORD) / sizeof(STOP_WORD[0])),
        .pieces = SERVER_STOP_PIECES,
        .piece_count = (int)(sizeof(SERVER_STOP_PIECES) / sizeof(SERVER_STOP_PIECES[0])),
    },
    {
        .name = "server_openai_stream_holds_stop_tail",
        .source = "ds4_server.c:stop_list_stream_safe_len",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .stream = true,
        .max_tokens = 4,
        .stops = STOP_END,
        .stop_count = (int)(sizeof(STOP_END) / sizeof(STOP_END[0])),
        .pieces = SERVER_STOP_HOLD_PIECES,
        .piece_count = (int)(sizeof(SERVER_STOP_HOLD_PIECES) / sizeof(SERVER_STOP_HOLD_PIECES[0])),
    },
    {
        .name = "server_openai_stream_stop_hit_discards_tail",
        .source = "ds4_server.c:generate_job streaming stop hit",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .stream = true,
        .max_tokens = 4,
        .stops = STOP_WORD,
        .stop_count = (int)(sizeof(STOP_WORD) / sizeof(STOP_WORD[0])),
        .pieces = SERVER_STREAM_STOP_HIT_PIECES,
        .piece_count = (int)(sizeof(SERVER_STREAM_STOP_HIT_PIECES) / sizeof(SERVER_STREAM_STOP_HIT_PIECES[0])),
    },
    {
        .name = "server_openai_stream_holds_partial_utf8",
        .source = "ds4_server.c:utf8_stream_safe_len",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .stream = true,
        .max_tokens = 4,
        .pieces = SERVER_UTF8_HOLD_PIECES,
        .piece_count = (int)(sizeof(SERVER_UTF8_HOLD_PIECES) / sizeof(SERVER_UTF8_HOLD_PIECES[0])),
    },
    {
        .name = "server_openai_stop_mid_utf8_boundary",
        .source = "ds4_server.c:utf8_stream_safe_len hit_stop",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .stream = true,
        .max_tokens = 4,
        .stops = STOP_WORD,
        .stop_count = (int)(sizeof(STOP_WORD) / sizeof(STOP_WORD[0])),
        .pieces = SERVER_STOP_MID_UTF8_PIECES,
        .piece_count = (int)(sizeof(SERVER_STOP_MID_UTF8_PIECES) / sizeof(SERVER_STOP_MID_UTF8_PIECES[0])),
    },
    {
        .name = "server_openai_tool_call_boundary",
        .source = "ds4_server.c:observe_tool_markers tool_calls",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .has_tools = true,
        .max_tokens = 8,
        .pieces = SERVER_TOOL_PIECES,
        .piece_count = (int)(sizeof(SERVER_TOOL_PIECES) / sizeof(SERVER_TOOL_PIECES[0])),
    },
    {
        .name = "server_responses_length_mapping",
        .source = "ds4_server.c:responses_status_for_finish length",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_RESPONSES,
        .kind = REQ_CHAT,
        .max_tokens = 2,
        .pieces = SERVER_LENGTH_PIECES,
        .piece_count = (int)(sizeof(SERVER_LENGTH_PIECES) / sizeof(SERVER_LENGTH_PIECES[0])),
    },
    {
        .name = "server_anthropic_tool_mapping",
        .source = "ds4_server.c:anthropic_stop_reason tool_calls",
        .surface = POLICY_SURFACE_SERVER,
        .api = API_ANTHROPIC,
        .kind = REQ_CHAT,
        .has_tools = true,
        .max_tokens = 8,
        .pieces = SERVER_TOOL_PIECES,
        .piece_count = (int)(sizeof(SERVER_TOOL_PIECES) / sizeof(SERVER_TOOL_PIECES[0])),
    },
    {
        .name = "agent_eos_stop",
        .source = "ds4_agent.c:agent_worker_run EOS",
        .surface = POLICY_SURFACE_AGENT,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .max_tokens = 4,
        .pieces = AGENT_EOS_PIECES,
        .piece_count = (int)(sizeof(AGENT_EOS_PIECES) / sizeof(AGENT_EOS_PIECES[0])),
    },
    {
        .name = "agent_max_tokens_length",
        .source = "ds4_agent.c:agent_worker_run max_tokens",
        .surface = POLICY_SURFACE_AGENT,
        .api = API_OPENAI,
        .kind = REQ_CHAT,
        .max_tokens = 2,
        .pieces = AGENT_LENGTH_PIECES,
        .piece_count = (int)(sizeof(AGENT_LENGTH_PIECES) / sizeof(AGENT_LENGTH_PIECES[0])),
    },
};

static const char *policy_surface_name(policy_surface surface) {
    switch (surface) {
    case POLICY_SURFACE_CLI: return "cli";
    case POLICY_SURFACE_SERVER: return "server";
    case POLICY_SURFACE_AGENT: return "agent";
    }
    return "unknown";
}

static const char *policy_api_name(api_style api) {
    switch (api) {
    case API_OPENAI: return "openai";
    case API_ANTHROPIC: return "anthropic";
    case API_RESPONSES: return "responses";
    }
    return "unknown";
}

static const char *policy_kind_name(req_kind kind) {
    return kind == REQ_CHAT ? "chat" : "completion";
}

static void policy_json_bool(FILE *fp, bool value) {
    fputs(value ? "true" : "false", fp);
}

static void policy_json_string(FILE *fp, const char *s) {
    buf b = {0};
    json_escape(&b, s ? s : "");
    fputs(b.ptr ? b.ptr : "\"\"", fp);
    buf_free(&b);
}

static void policy_json_null_or_string(FILE *fp, const char *s) {
    if (s) policy_json_string(fp, s);
    else fputs("null", fp);
}

static void policy_hex_buf(buf *b, const char *s, size_t n) {
    static const char hex[] = "0123456789abcdef";
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        buf_putc(b, hex[c >> 4]);
        buf_putc(b, hex[c & 0x0f]);
    }
    buf_putc(b, '"');
}

static void policy_hex_file(FILE *fp, const char *s, size_t n) {
    buf b = {0};
    policy_hex_buf(&b, s ? s : "", s ? n : 0);
    fputs(b.ptr ? b.ptr : "\"\"", fp);
    buf_free(&b);
}

static void policy_append_stream_step(buf *steps, bool *first, int index,
                                      size_t text_len, size_t stream_len,
                                      const char *delta, size_t delta_len,
                                      const char *held, size_t held_len,
                                      bool hit_stop, size_t stop_pos,
                                      size_t stop_len) {
    if (!*first) buf_puts(steps, ",");
    *first = false;
    buf_printf(steps,
               "{\"step\":%d,\"text_len\":%zu,\"stream_safe_len\":%zu,"
               "\"delta_hex\":",
               index, text_len, stream_len);
    policy_hex_buf(steps, delta, delta_len);
    buf_puts(steps, ",\"held_tail_hex\":");
    policy_hex_buf(steps, held, held_len);
    buf_puts(steps, ",\"hit_stop\":");
    buf_puts(steps, hit_stop ? "true" : "false");
    buf_printf(steps, ",\"stop_pos\":%zd,\"stop_len\":%zu}",
               hit_stop ? (ssize_t)stop_pos : (ssize_t)-1, stop_len);
}

static void policy_stops_init(stop_list *stops, const policy_case *tc) {
    memset(stops, 0, sizeof(*stops));
    for (int i = 0; i < tc->stop_count; i++) {
        stop_list_push(stops, xstrdup(tc->stops[i]));
    }
}

static const char *policy_responses_incomplete_reason(const char *finish) {
    return finish && !strcmp(finish, "length") ? "max_tokens" : NULL;
}

static void policy_result_free(policy_result *r) {
    buf_free(&r->raw_text);
    buf_free(&r->visible_text);
    buf_free(&r->reasoning_text);
    buf_free(&r->streamed_text);
    buf_free(&r->stream_steps);
    free(r->finish);
    memset(r, 0, sizeof(*r));
}

static void policy_run_server_case(const policy_case *tc, policy_result *out) {
    request r;
    request_init(&r, tc->kind, tc->max_tokens);
    r.api = tc->api;
    r.stream = tc->stream;
    r.has_tools = tc->has_tools;
    r.think_mode = DS4_THINK_NONE;
    policy_stops_init(&r.stops, tc);

    const char *finish = "length";
    int completion = 0;
    size_t plain_stream_pos = 0;
    size_t stop_scan_from = 0;
    size_t tool_scan_from = 0;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool saw_orphan_tool_end = false;
    bool first_step = true;

    for (int i = 0; i < tc->piece_count && completion < r.max_tokens; i++) {
        const policy_piece *piece = &tc->pieces[i];
        if (piece->eos) {
            finish = "stop";
            break;
        }

        size_t piece_len = strlen(piece->text);
        buf_append(&out->raw_text, piece->text, piece_len);
        completion++;

        if (r.kind == REQ_CHAT && r.has_tools) {
            if (tool_scan_from > out->raw_text.len) tool_scan_from = out->raw_text.len;
            const char *tool_scan = out->raw_text.ptr ? out->raw_text.ptr + tool_scan_from : "";
            observe_tool_markers(tool_scan, &saw_tool_start, &saw_tool_end, &saw_orphan_tool_end);
            const size_t marker_hold = 80;
            size_t hold_from = out->raw_text.len > marker_hold ? out->raw_text.len - marker_hold : 0;
            if (hold_from > tool_scan_from) tool_scan_from = hold_from;
        }

        size_t stop_pos = 0;
        size_t stop_len = 0;
        bool hit_stop = stop_list_find_from(&r.stops, out->raw_text.ptr,
                                            stop_scan_from, &stop_pos, &stop_len);
        size_t stream_len = hit_stop ?
            stop_pos : stop_list_stream_safe_len(&r.stops, out->raw_text.len);
        if (stream_len > out->raw_text.len) stream_len = out->raw_text.len;
        stream_len = utf8_stream_safe_len(out->raw_text.ptr, plain_stream_pos,
                                          stream_len, hit_stop);
        if (!hit_stop && r.stops.max_len > 1) {
            const size_t hold = r.stops.max_len - 1;
            stop_scan_from = out->raw_text.len > hold ? out->raw_text.len - hold : 0;
        }

        const char *delta = "";
        size_t delta_len = 0;
        if (r.stream && stream_len > plain_stream_pos) {
            delta = out->raw_text.ptr + plain_stream_pos;
            delta_len = stream_len - plain_stream_pos;
            buf_append(&out->streamed_text, delta, delta_len);
            plain_stream_pos = stream_len;
        }
        const char *held = out->raw_text.ptr ? out->raw_text.ptr + plain_stream_pos : "";
        size_t held_len = out->raw_text.len >= plain_stream_pos ?
            out->raw_text.len - plain_stream_pos : 0;
        if (!r.stream) {
            held = "";
            held_len = 0;
        }
        if (hit_stop) {
            held = "";
            held_len = 0;
        }
        policy_append_stream_step(&out->stream_steps, &first_step, i,
                                  out->raw_text.len, stream_len,
                                  delta, delta_len, held, held_len,
                                  hit_stop, stop_pos, stop_len);

        if (hit_stop) {
            finish = "stop";
            out->raw_text.len = stop_pos;
            if (out->raw_text.ptr) out->raw_text.ptr[out->raw_text.len] = '\0';
            out->session_invalidated = true;
            out->stop_pos = (int)stop_pos;
            out->stop_len = (int)stop_len;
            break;
        }
        if (r.kind == REQ_CHAT && r.has_tools && saw_tool_end) {
            finish = "tool_calls";
            break;
        }
    }

    if (r.stream && out->raw_text.len > plain_stream_pos) {
        buf_append(&out->streamed_text,
                   out->raw_text.ptr + plain_stream_pos,
                   out->raw_text.len - plain_stream_pos);
    }

    char err[160] = {0};
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool recovered = false;
    const char *final_finish = finish;
    if (r.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            out->raw_text.ptr ? out->raw_text.ptr : "",
            r.has_tools,
            saw_tool_start,
            ds4_think_mode_enabled(r.think_mode),
            &final_finish,
            err,
            sizeof(err),
            &content,
            &reasoning,
            &calls,
            &recovered);
        (void)parsed_ok;
        (void)recovered;
        if (calls.len) final_finish = "tool_calls";
    }

    out->finish = xstrdup(final_finish);
    out->completion_tokens = completion;
    out->saw_tool_start = saw_tool_start;
    out->saw_tool_end = saw_tool_end;
    out->tool_call_count = calls.len;
    if (content) buf_puts(&out->visible_text, content);
    else buf_puts(&out->visible_text, out->raw_text.ptr ? out->raw_text.ptr : "");
    if (reasoning) buf_puts(&out->reasoning_text, reasoning);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    request_free(&r);
}

static void policy_run_simple_case(const policy_case *tc, policy_result *out) {
    const char *finish = "length";
    int generated = 0;
    bool first_step = true;
    for (int i = 0; i < tc->piece_count && generated < tc->max_tokens; i++) {
        const policy_piece *piece = &tc->pieces[i];
        if (piece->eos) {
            finish = "stop";
            break;
        }
        size_t n = strlen(piece->text);
        buf_append(&out->raw_text, piece->text, n);
        buf_append(&out->visible_text, piece->text, n);
        generated++;
        policy_append_stream_step(&out->stream_steps, &first_step, i,
                                  out->raw_text.len, out->raw_text.len,
                                  piece->text, n, "", 0, false, 0, 0);
    }
    out->finish = xstrdup(finish);
    out->completion_tokens = generated;
    out->transcript_eos_appended = tc->surface == POLICY_SURFACE_AGENT;
}

static void policy_run_case(const policy_case *tc, policy_result *out) {
    memset(out, 0, sizeof(*out));
    out->stop_pos = -1;
    out->stop_len = 0;
    if (tc->surface == POLICY_SURFACE_SERVER) policy_run_server_case(tc, out);
    else policy_run_simple_case(tc, out);
}

static void policy_write_schedule(FILE *fp, const policy_case *tc) {
    fputs("[", fp);
    for (int i = 0; i < tc->piece_count; i++) {
        if (i) fputs(",", fp);
        const policy_piece *piece = &tc->pieces[i];
        fprintf(fp, "{\"index\":%d,\"eos\":", i);
        policy_json_bool(fp, piece->eos);
        fputs(",\"text_hex\":", fp);
        policy_hex_file(fp, piece->text, piece->text ? strlen(piece->text) : 0);
        fputs("}", fp);
    }
    fputs("]", fp);
}

static void policy_write_stops(FILE *fp, const policy_case *tc) {
    fputs("[", fp);
    for (int i = 0; i < tc->stop_count; i++) {
        if (i) fputs(",", fp);
        policy_json_string(fp, tc->stops[i]);
    }
    fputs("]", fp);
}

static void policy_write_case(FILE *fp, const policy_case *tc, bool first) {
    policy_result r;
    policy_run_case(tc, &r);

    if (!first) fputs(",\n", fp);
    fputs("    {\"name\":", fp);
    policy_json_string(fp, tc->name);
    fputs(",\"source\":", fp);
    policy_json_string(fp, tc->source);
    fputs(",\"request\":{\"surface\":", fp);
    policy_json_string(fp, policy_surface_name(tc->surface));
    fputs(",\"api\":", fp);
    policy_json_string(fp, policy_api_name(tc->api));
    fputs(",\"kind\":", fp);
    policy_json_string(fp, policy_kind_name(tc->kind));
    fprintf(fp, ",\"stream\":");
    policy_json_bool(fp, tc->stream);
    fprintf(fp, ",\"has_tools\":");
    policy_json_bool(fp, tc->has_tools);
    fprintf(fp, ",\"max_tokens\":%d,\"stops\":", tc->max_tokens);
    policy_write_stops(fp, tc);
    fputs("},\"schedule\":", fp);
    policy_write_schedule(fp, tc);
    fputs(",\"result\":{\"finish_reason\":", fp);
    policy_json_string(fp, r.finish ? r.finish : "");
    fprintf(fp, ",\"completion_tokens\":%d", r.completion_tokens);
    fputs(",\"raw_text_hex\":", fp);
    policy_hex_file(fp, r.raw_text.ptr, r.raw_text.len);
    fputs(",\"visible_text_hex\":", fp);
    policy_hex_file(fp, r.visible_text.ptr, r.visible_text.len);
    fputs(",\"reasoning_hex\":", fp);
    policy_hex_file(fp, r.reasoning_text.ptr, r.reasoning_text.len);
    fputs(",\"streamed_text_hex\":", fp);
    policy_hex_file(fp, r.streamed_text.ptr, r.streamed_text.len);
    fprintf(fp, ",\"session_invalidation_required\":");
    policy_json_bool(fp, r.session_invalidated);
    fprintf(fp, ",\"transcript_eos_appended\":");
    policy_json_bool(fp, r.transcript_eos_appended);
    fprintf(fp, ",\"stop_boundary\":{\"pos\":%d,\"len\":%d}", r.stop_pos, r.stop_len);
    fprintf(fp, ",\"tool_boundary\":{\"saw_start\":");
    policy_json_bool(fp, r.saw_tool_start);
    fprintf(fp, ",\"saw_end\":");
    policy_json_bool(fp, r.saw_tool_end);
    fprintf(fp, ",\"tool_call_count\":%d}", r.tool_call_count);
    fputs(",\"api_finish\":{\"openai_finish_reason\":", fp);
    if (tc->surface == POLICY_SURFACE_SERVER) policy_json_string(fp, r.finish ? r.finish : "");
    else fputs("null", fp);
    fputs(",\"anthropic_stop_reason\":", fp);
    if (tc->surface == POLICY_SURFACE_SERVER)
        policy_json_string(fp, anthropic_stop_reason(r.finish));
    else fputs("null", fp);
    fputs(",\"responses_status\":", fp);
    if (tc->surface == POLICY_SURFACE_SERVER)
        policy_json_string(fp, responses_status_for_finish(r.finish));
    else fputs("null", fp);
    fputs(",\"responses_item_status\":", fp);
    if (tc->surface == POLICY_SURFACE_SERVER)
        policy_json_string(fp, responses_item_status_for_finish(r.finish));
    else fputs("null", fp);
    fputs(",\"responses_incomplete_reason\":", fp);
    if (tc->surface == POLICY_SURFACE_SERVER)
        policy_json_null_or_string(fp, policy_responses_incomplete_reason(r.finish));
    else fputs("null", fp);
    fputs("},\"stream_steps\":[", fp);
    fputs(r.stream_steps.ptr ? r.stream_steps.ptr : "", fp);
    fputs("]}}", fp);

    policy_result_free(&r);
}

static int policy_dump_json(FILE *fp) {
    fputs("{\n", fp);
    fputs("  \"schema\": \"ds4.decode_policy_oracle.v1\",\n", fp);
    fputs("  \"source\": \"current-c-decode-stop-policy\",\n", fp);
    fputs("  \"model\": \"no model is loaded for this oracle\",\n", fp);
    fputs("  \"cases\": [\n", fp);
    for (size_t i = 0; i < sizeof(POLICY_CASES) / sizeof(POLICY_CASES[0]); i++) {
        policy_write_case(fp, &POLICY_CASES[i], i == 0);
    }
    fputs("\n  ]\n", fp);
    fputs("}\n", fp);
    return ferror(fp) ? 1 : 0;
}

static void policy_usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4-decode-policy-dump [OUTPUT]\n"
            "\n"
            "Emit deterministic no-model JSON for current-C decode stop policy.\n"
            "\n"
            "Arguments:\n"
            "  OUTPUT   Optional output file. Defaults to stdout.\n"
            "  -h, --help\n"
            "           Show this help\n");
}

int main(int argc, char **argv) {
    const char *output_path = NULL;
    if (argc > 2) {
        policy_usage(stderr);
        return 2;
    }
    if (argc == 2) {
        if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
            policy_usage(stdout);
            return 0;
        }
        output_path = argv[1];
    }

    FILE *fp = stdout;
    if (output_path) {
        fp = fopen(output_path, "wb");
        if (!fp) {
            fprintf(stderr, "ds4-decode-policy-dump: failed to open %s: %s\n",
                    output_path, strerror(errno));
            return 1;
        }
    }

    int rc = policy_dump_json(fp);
    if (output_path && fclose(fp) != 0) {
        fprintf(stderr, "ds4-decode-policy-dump: failed to close %s: %s\n",
                output_path, strerror(errno));
        return 1;
    }
    return rc;
}
