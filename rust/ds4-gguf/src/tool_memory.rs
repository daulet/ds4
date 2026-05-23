use std::collections::{HashMap, VecDeque};

use crate::prompt::ChatMessage;

pub const TOOL_MEMORY_DEFAULT_MAX_IDS: usize = 100_000;
pub const TOOL_MEMORY_MAX_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolMemorySource {
    Ram,
    Disk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolReplayStats {
    pub mem: usize,
    pub disk: usize,
    pub canonical: usize,
    pub missing_ids: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolMemoryEntry {
    dsml: String,
    source: ToolMemorySource,
    bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolMemory {
    by_id: HashMap<String, ToolMemoryEntry>,
    recent: VecDeque<String>,
    entries: usize,
    max_entries: usize,
    bytes: usize,
    max_bytes: usize,
}

impl Default for ToolMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolMemory {
    pub fn new() -> Self {
        Self::with_limits(TOOL_MEMORY_DEFAULT_MAX_IDS, TOOL_MEMORY_MAX_BYTES)
    }

    pub fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            by_id: HashMap::new(),
            recent: VecDeque::new(),
            entries: 0,
            max_entries: max_entries.max(1),
            bytes: 0,
            max_bytes: max_bytes.max(1),
        }
    }

    pub fn entries(&self) -> usize {
        self.entries
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn contains_id(&self, id: &str) -> bool {
        !id.is_empty() && self.by_id.contains_key(id)
    }

    pub fn put_ram(&mut self, id: &str, dsml: &str) {
        self.put_source(id, dsml, ToolMemorySource::Ram);
    }

    pub fn put_disk(&mut self, id: &str, dsml: &str) {
        self.put_source(id, dsml, ToolMemorySource::Disk);
    }

    pub fn put_source(&mut self, id: &str, dsml: &str, source: ToolMemorySource) {
        if id.is_empty() || dsml.is_empty() {
            return;
        }

        if let Some(old) = self.by_id.get_mut(id) {
            if old.dsml == dsml {
                if source == ToolMemorySource::Ram {
                    old.source = ToolMemorySource::Ram;
                }
                self.touch(id);
                self.prune();
                return;
            }
        }
        self.remove(id);

        let bytes = entry_bytes(id, dsml);
        self.by_id.insert(
            id.to_string(),
            ToolMemoryEntry {
                dsml: dsml.to_string(),
                source,
                bytes,
            },
        );
        self.recent.push_front(id.to_string());
        self.entries += 1;
        self.bytes = self.bytes.saturating_add(bytes);
        self.prune();
    }

    pub fn remember_ids<'a>(
        &mut self,
        ids: impl IntoIterator<Item = &'a str>,
        dsml: &str,
        source: ToolMemorySource,
    ) {
        if dsml.is_empty() {
            return;
        }
        for id in ids {
            self.put_source(id, dsml, source);
        }
    }

    pub fn attach_to_messages(&mut self, messages: &mut [ChatMessage]) -> ToolReplayStats {
        let mut stats = ToolReplayStats::default();
        for message in messages {
            if message.tool_calls.is_empty() || message.raw_tool_calls_dsml.is_some() {
                continue;
            }

            let mut matched_dsml: Option<String> = None;
            let mut matched_source = ToolMemorySource::Disk;
            let mut exact = true;
            let mut missing = 0;

            for call in &message.tool_calls {
                let id = call.id.as_str();
                let Some((dsml, source)) = self.lookup(id) else {
                    exact = false;
                    missing += 1;
                    continue;
                };
                match &matched_dsml {
                    Some(matched) if matched != &dsml => exact = false,
                    None => {
                        matched_dsml = Some(dsml);
                        matched_source = source;
                    }
                    _ => {}
                }
                if source == ToolMemorySource::Ram {
                    matched_source = ToolMemorySource::Ram;
                }
            }

            if exact {
                if let Some(dsml) = matched_dsml {
                    message.raw_tool_calls_dsml = Some(dsml);
                    match matched_source {
                        ToolMemorySource::Ram => stats.mem += 1,
                        ToolMemorySource::Disk => stats.disk += 1,
                    }
                    continue;
                }
            }

            stats.canonical += 1;
            stats.missing_ids += missing;
        }
        stats
    }

    fn lookup(&mut self, id: &str) -> Option<(String, ToolMemorySource)> {
        let entry = self.by_id.get(id)?;
        let dsml = entry.dsml.clone();
        let source = entry.source;
        self.touch(id);
        Some((dsml, source))
    }

    fn touch(&mut self, id: &str) {
        if id.is_empty() {
            return;
        }
        self.recent.retain(|existing| existing != id);
        self.recent.push_front(id.to_string());
    }

    fn remove(&mut self, id: &str) {
        if let Some(old) = self.by_id.remove(id) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
            self.entries = self.entries.saturating_sub(1);
        }
        self.recent.retain(|existing| existing != id);
    }

    fn prune(&mut self) {
        while (self.entries > self.max_entries || self.bytes > self.max_bytes)
            && !self.recent.is_empty()
        {
            if let Some(id) = self.recent.pop_back() {
                if let Some(old) = self.by_id.remove(&id) {
                    self.bytes = self.bytes.saturating_sub(old.bytes);
                    self.entries = self.entries.saturating_sub(1);
                }
            }
        }
    }
}

fn entry_bytes(id: &str, dsml: &str) -> usize {
    id.len()
        .saturating_add(1)
        .saturating_add(dsml.len())
        .saturating_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        parse_generated_message, render_chat_prompt_text, ChatMessage, ThinkMode, ToolArgument,
        ToolCall,
    };

    fn sampled_openai_dsml() -> &'static str {
        "<think>need shell</think>\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"bash\">\n\
<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>"
    }

    fn bash_call(id: &str, args: Vec<ToolArgument>) -> ToolCall {
        ToolCall::new("bash", args).with_id(id)
    }

    #[test]
    fn tool_memory_replays_sampled_dsml_before_prompt_render() {
        let sampled = parse_generated_message(sampled_openai_dsml(), false).expect("parse DSML");
        let raw_dsml = sampled.raw_dsml.as_deref().expect("raw DSML");
        let mut memory = ToolMemory::new();
        memory.remember_ids(["call_exact"], raw_dsml, ToolMemorySource::Ram);

        let mut messages = vec![
            ChatMessage::new("assistant", sampled.content).with_tool_calls(vec![bash_call(
                "call_exact",
                vec![
                    ToolArgument::string("description", "list files"),
                    ToolArgument::string("command", "ls -la"),
                    ToolArgument {
                        name: "timeout".to_string(),
                        value: "10".to_string(),
                        is_string: false,
                    },
                ],
            )]),
        ];
        messages[0].reasoning = sampled.reasoning.unwrap_or_default();

        let stats = memory.attach_to_messages(&mut messages);
        assert_eq!(
            stats,
            ToolReplayStats {
                mem: 1,
                disk: 0,
                canonical: 0,
                missing_ids: 0,
            }
        );
        let prompt = render_chat_prompt_text(&messages, None, ThinkMode::High);
        let command = prompt.find("name=\"command\"").expect("command");
        let timeout = prompt.find("name=\"timeout\"").expect("timeout");
        let description = prompt.find("name=\"description\"").expect("description");
        assert!(command < timeout);
        assert!(timeout < description);
    }

    #[test]
    fn anthropic_tool_memory_replays_sampled_dsml() {
        let sampled_dsml = "\n\n<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"Bash\">\n\
<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";
        let mut memory = ToolMemory::new();
        memory.put_ram("toolu_exact", sampled_dsml);

        let mut tool_result = ChatMessage::new("user", "<tool_result>ok</tool_result>");
        tool_result.add_tool_call_id("toolu_exact");
        let mut messages = vec![
            ChatMessage::new("assistant", "").with_tool_calls(vec![ToolCall::new(
                "Bash",
                vec![
                    ToolArgument::string("description", "list files"),
                    ToolArgument::string("command", "ls -la"),
                ],
            )
            .with_id("toolu_exact")]),
            tool_result,
        ];

        let stats = memory.attach_to_messages(&mut messages);
        assert_eq!(stats.mem, 1);
        assert_eq!(stats.canonical, 0);
        let prompt = render_chat_prompt_text(&messages, None, ThinkMode::High);
        let command = prompt.find("name=\"command\"").expect("command");
        let description = prompt.find("name=\"description\"").expect("description");
        assert!(command < description);
    }

    #[test]
    fn tool_memory_uses_canonical_render_when_ids_are_missing_or_split() {
        let mut memory = ToolMemory::new();
        memory.put_disk(
            "call_keep",
            "\n\n<tool_calls><invoke name=\"bash\"></invoke></tool_calls>",
        );
        memory.put_ram(
            "call_a",
            "\n\n<tool_calls><invoke name=\"a\"></invoke></tool_calls>",
        );
        memory.put_ram(
            "call_b",
            "\n\n<tool_calls><invoke name=\"b\"></invoke></tool_calls>",
        );

        let mut messages = vec![
            ChatMessage::new("assistant", "").with_tool_calls(vec![bash_call(
                "call_keep",
                vec![ToolArgument::string("command", "pwd")],
            )]),
            ChatMessage::new("assistant", "").with_tool_calls(vec![bash_call(
                "call_missing",
                vec![ToolArgument::string("command", "whoami")],
            )]),
            ChatMessage::new("assistant", "").with_tool_calls(vec![
                ToolCall::new("a", Vec::new()).with_id("call_a"),
                ToolCall::new("b", Vec::new()).with_id("call_b"),
            ]),
        ];

        let stats = memory.attach_to_messages(&mut messages);
        assert_eq!(
            stats,
            ToolReplayStats {
                mem: 0,
                disk: 1,
                canonical: 2,
                missing_ids: 1,
            }
        );
        assert!(messages[0].raw_tool_calls_dsml.is_some());
        assert!(messages[1].raw_tool_calls_dsml.is_none());
        assert!(messages[2].raw_tool_calls_dsml.is_none());
    }

    #[test]
    fn tool_memory_prunes_oldest_ids_and_touches_lookup() {
        let mut memory = ToolMemory::with_limits(2, TOOL_MEMORY_MAX_BYTES);
        memory.put_ram("call_a", "dsml-a");
        memory.put_ram("call_b", "dsml-b");
        memory.put_ram("call_c", "dsml-c");
        assert!(!memory.contains_id("call_a"));
        assert!(memory.contains_id("call_b"));
        assert!(memory.contains_id("call_c"));

        let mut messages = vec![ChatMessage::new("assistant", "")
            .with_tool_calls(vec![ToolCall::new("b", Vec::new()).with_id("call_b")])];
        let stats = memory.attach_to_messages(&mut messages);
        assert_eq!(stats.mem, 1);

        memory.put_ram("call_d", "dsml-d");
        assert!(memory.contains_id("call_b"));
        assert!(!memory.contains_id("call_c"));
        assert!(memory.contains_id("call_d"));
        assert_eq!(memory.entries(), 2);
    }

    #[test]
    fn tool_memory_upgrades_disk_entry_to_ram_without_duplicate() {
        let mut memory = ToolMemory::new();
        memory.put_disk("call_same", "dsml");
        memory.put_ram("call_same", "dsml");
        assert_eq!(memory.entries(), 1);

        let mut messages = vec![ChatMessage::new("assistant", "")
            .with_tool_calls(vec![ToolCall::new("tool", Vec::new()).with_id("call_same")])];
        let stats = memory.attach_to_messages(&mut messages);
        assert_eq!(stats.mem, 1);
        assert_eq!(stats.disk, 0);
    }
}
