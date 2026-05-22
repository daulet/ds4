const START: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const INVOKE_START: &[u8] = "<｜DSML｜invoke".as_bytes();
const PARAM_START: &[u8] = "<｜DSML｜parameter".as_bytes();
const CLOSE_PREFIX: &[u8] = "</｜DSML｜".as_bytes();
const DSML_BAR: &[u8] = "｜".as_bytes();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentDsmlState {
    Search,
    Structural,
    ParamValue,
    Done,
    Error,
}

impl AgentDsmlState {
    pub fn name(self) -> &'static str {
        match self {
            Self::Search => "search",
            Self::Structural => "structural",
            Self::ParamValue => "param_value",
            Self::Done => "done",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgentToolArg {
    pub name: String,
    pub value: String,
    pub is_string: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgentToolCall {
    pub name: Option<String>,
    pub args: Vec<AgentToolArg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentDsmlParser {
    pub state: AgentDsmlState,
    pub search_tail: Vec<u8>,
    pub raw: Vec<u8>,
    pub parse_pos: usize,
    pub current: AgentToolCall,
    pub param_name: Option<String>,
    pub param_is_string: bool,
    pub param_value_start: usize,
    pub calls: Vec<AgentToolCall>,
    pub error: String,
}

impl Default for AgentDsmlParser {
    fn default() -> Self {
        Self {
            state: AgentDsmlState::Search,
            search_tail: Vec::new(),
            raw: Vec::new(),
            parse_pos: 0,
            current: AgentToolCall::default(),
            param_name: None,
            param_is_string: false,
            param_value_start: 0,
            calls: Vec::new(),
            error: String::new(),
        }
    }
}

impl AgentDsmlParser {
    pub fn feed(&mut self, bytes: &[u8]) {
        if matches!(self.state, AgentDsmlState::Done | AgentDsmlState::Error) {
            return;
        }
        for &byte in bytes {
            if self.state == AgentDsmlState::Search {
                if self.search_tail.len() == 64 {
                    self.search_tail.remove(0);
                }
                self.search_tail.push(byte);
                if self.search_tail.len() >= START.len()
                    && &self.search_tail[self.search_tail.len() - START.len()..] == START
                {
                    self.start();
                }
                continue;
            }

            self.raw.push(byte);
            self.parse();
        }
    }

    fn start(&mut self) {
        self.state = AgentDsmlState::Structural;
        self.search_tail.clear();
        self.raw.extend_from_slice(START);
        self.parse_pos = START.len();
    }

    fn parse(&mut self) {
        while matches!(
            self.state,
            AgentDsmlState::Structural | AgentDsmlState::ParamValue
        ) {
            if self.state == AgentDsmlState::ParamValue {
                let Some((end, end_tag_len)) =
                    find_close_tag(&self.raw[self.param_value_start..], b"parameter")
                else {
                    return;
                };
                let end = self.param_value_start + end;
                let name = self.param_name.clone().unwrap_or_default();
                let value =
                    String::from_utf8_lossy(&self.raw[self.param_value_start..end]).into_owned();
                self.current.args.push(AgentToolArg {
                    name,
                    value,
                    is_string: self.param_is_string,
                });
                self.param_name = None;
                self.parse_pos = end + end_tag_len;
                self.state = AgentDsmlState::Structural;
                continue;
            }

            while self.parse_pos < self.raw.len()
                && matches!(self.raw[self.parse_pos], b' ' | b'\t' | b'\r' | b'\n')
            {
                self.parse_pos += 1;
            }
            if self.parse_pos >= self.raw.len() {
                return;
            }

            if let Some(close_len) = close_tag_at(&self.raw[self.parse_pos..], b"tool_calls") {
                self.push_current();
                self.parse_pos += close_len;
                self.state = AgentDsmlState::Done;
                return;
            }
            if let Some(close_len) = close_tag_at(&self.raw[self.parse_pos..], b"invoke") {
                self.push_current();
                self.parse_pos += close_len;
                continue;
            }

            let Some(tag_end_rel) = self.raw[self.parse_pos..]
                .iter()
                .position(|&byte| byte == b'>')
            else {
                return;
            };
            let tag_len = tag_end_rel + 1;
            let tag = &self.raw[self.parse_pos..self.parse_pos + tag_len];

            if tag.starts_with(INVOKE_START) {
                self.current = AgentToolCall::default();
                self.current.name = parse_attr(tag, b"name");
                if self.current.name.is_none() {
                    self.set_error("tool invoke without name");
                    return;
                }
                self.parse_pos += tag_len;
            } else if tag.starts_with(PARAM_START) {
                self.param_name = parse_attr(tag, b"name");
                let is_string = parse_attr(tag, b"string");
                self.param_is_string = is_string.as_deref() == Some("true");
                if self.param_name.is_none() {
                    self.set_error("tool parameter without name");
                    return;
                }
                self.parse_pos += tag_len;
                self.param_value_start = self.parse_pos;
                self.state = AgentDsmlState::ParamValue;
            } else {
                let shown_len = tag_len.min(80);
                let shown = String::from_utf8_lossy(&tag[..shown_len]);
                self.error = format!("unexpected DSML tag: {shown}");
                self.state = AgentDsmlState::Error;
                return;
            }
        }
    }

    fn push_current(&mut self) {
        if self.current.name.is_some() {
            self.calls.push(std::mem::take(&mut self.current));
        }
    }

    fn set_error(&mut self, message: &str) {
        self.state = AgentDsmlState::Error;
        self.error.clear();
        self.error.push_str(message);
    }
}

fn close_tag_at(s: &[u8], name: &[u8]) -> Option<usize> {
    if !s.starts_with(CLOSE_PREFIX) {
        return None;
    }
    let mut pos = CLOSE_PREFIX.len();
    if s.get(pos..pos + name.len())? != name {
        return None;
    }
    pos += name.len();
    while matches!(s.get(pos), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        pos += 1;
    }
    if s.get(pos..pos + DSML_BAR.len()) == Some(DSML_BAR) {
        pos += DSML_BAR.len();
    }
    while matches!(s.get(pos), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        pos += 1;
    }
    if s.get(pos) != Some(&b'>') {
        return None;
    }
    Some(pos + 1)
}

fn find_close_tag(s: &[u8], name: &[u8]) -> Option<(usize, usize)> {
    let mut pos = 0;
    while pos + CLOSE_PREFIX.len() <= s.len() {
        let hay = &s[pos..];
        let Some(rel) = find_bytes(hay, CLOSE_PREFIX) else {
            return None;
        };
        pos += rel;
        if let Some(tag_len) = close_tag_at(&s[pos..], name) {
            return Some((pos, tag_len));
        }
        pos += 1;
    }
    None
}

fn parse_attr(tag: &[u8], name: &[u8]) -> Option<String> {
    let mut pattern = Vec::with_capacity(name.len() + 2);
    pattern.extend_from_slice(name);
    pattern.extend_from_slice(b"=\"");
    let start = find_bytes(tag, &pattern)? + pattern.len();
    let end = tag[start..].iter().position(|&byte| byte == b'"')? + start;
    Some(String::from_utf8_lossy(&tag[start..end]).into_owned())
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_ignores_short_marker() {
        let mut parser = AgentDsmlParser::default();
        parser.feed(b"<DSML\xef\xbd\x9ctool_calls>");
        assert_eq!(parser.state, AgentDsmlState::Search);
        assert!(parser.raw.is_empty());
    }

    #[test]
    fn parser_accepts_close_tag_variants() {
        let mut parser = AgentDsmlParser::default();
        parser.feed(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter ｜ >\n</｜DSML｜invoke ｜ >\n</｜DSML｜tool_calls ｜ >"
                .as_bytes(),
        );
        assert_eq!(parser.state, AgentDsmlState::Done);
        assert_eq!(parser.calls.len(), 1);
        assert_eq!(parser.calls[0].args[0].value, "pwd");
    }
}
