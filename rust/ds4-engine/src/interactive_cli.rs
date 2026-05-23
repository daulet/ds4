use crate::ThinkMode;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplAction {
    Continue,
    Exit(i32),
    Quit,
    SetContext(i32),
    ReadFile(String),
    RunPrompt(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplCommandResult {
    pub action: ReplAction,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplState {
    pub think_mode: ThinkMode,
    pub ctx_size: i32,
}

impl ReplState {
    pub const fn new(think_mode: ThinkMode, ctx_size: i32) -> Self {
        Self {
            think_mode,
            ctx_size,
        }
    }

    pub fn handle_line(&mut self, line: &str) -> ReplCommandResult {
        let cmd = line.trim();
        if cmd.is_empty() {
            return result(ReplAction::Continue, "", "");
        }
        match cmd {
            "/help" => result(ReplAction::Continue, repl_help(), ""),
            "/think" => {
                self.think_mode = ThinkMode::High;
                result(ReplAction::Continue, "Thinking mode: high.\n", "")
            }
            "/think-max" => {
                self.think_mode = ThinkMode::Max;
                if self.think_mode.for_context(self.ctx_size) == ThinkMode::Max {
                    result(ReplAction::Continue, "Thinking mode: max.\n", "")
                } else {
                    result(
                        ReplAction::Continue,
                        "Thinking mode: high (ctx below 393216).\n",
                        &think_max_warning("/think-max", self.ctx_size),
                    )
                }
            }
            "/nothink" => {
                self.think_mode = ThinkMode::None;
                result(ReplAction::Continue, "Thinking mode: none.\n", "")
            }
            "/quit" | "/exit" => result(ReplAction::Quit, "", ""),
            _ if starts_command(cmd, "/ctx") => self.handle_ctx(cmd),
            _ if starts_command(cmd, "/read") => self.handle_read(cmd),
            _ if cmd.starts_with('/') => result(
                ReplAction::Continue,
                "",
                &format!("ds4: unknown command: {cmd}\nds4: type /help for commands\n"),
            ),
            _ => result(ReplAction::RunPrompt(cmd.to_string()), "", ""),
        }
    }

    pub fn handle_interrupt_at_prompt(&self) -> ReplCommandResult {
        result(ReplAction::Continue, "", "")
    }

    fn handle_ctx(&mut self, cmd: &str) -> ReplCommandResult {
        let value = cmd[4..].trim();
        if value.is_empty() {
            return result(
                ReplAction::Continue,
                "",
                "ds4: /ctx needs a positive integer\n",
            );
        }
        let Some(ctx_size) = parse_positive_i32(value) else {
            return result(
                ReplAction::Exit(2),
                "",
                &format!("ds4: invalid value for /ctx: {value}\n"),
            );
        };
        self.ctx_size = ctx_size;
        let stderr = if self.think_mode == ThinkMode::Max
            && self.think_mode.for_context(self.ctx_size) != ThinkMode::Max
        {
            think_max_warning("/ctx", self.ctx_size)
        } else {
            String::new()
        };
        result(ReplAction::SetContext(ctx_size), "", &stderr)
    }

    fn handle_read(&self, cmd: &str) -> ReplCommandResult {
        let path = cmd[5..].trim();
        if path.is_empty() {
            return result(ReplAction::Continue, "", "ds4: /read needs a file path\n");
        }
        result(ReplAction::ReadFile(path.to_string()), "", "")
    }
}

pub fn repl_help() -> &'static str {
    "Commands:\n\
  /help          Show this help.\n\
  /think         Use normal thinking mode.\n\
  /think-max     Use Think Max only when context is at least 393216 tokens.\n\
  /nothink       Disable thinking mode.\n\
  /ctx N         Set context size for following prompts.\n\
  /read FILE     Read a prompt from FILE and run it.\n\
  /quit, /exit   Leave the prompt.\n\
  Ctrl+C         Stop generation and return to the prompt.\n"
}

fn starts_command(cmd: &str, name: &str) -> bool {
    cmd == name
        || cmd
            .strip_prefix(name)
            .is_some_and(|rest| rest.starts_with(char::is_whitespace))
}

fn parse_positive_i32(value: &str) -> Option<i32> {
    let parsed = value.parse::<i64>().ok()?;
    if (1..=i32::MAX as i64).contains(&parsed) {
        Some(parsed as i32)
    } else {
        None
    }
}

fn think_max_warning(name: &str, ctx_size: i32) -> String {
    format!(
        "ds4: warning: {name} needs --ctx >= {}; ctx={ctx_size} uses normal thinking instead\n",
        ThinkMode::max_min_context()
    )
}

fn result(action: ReplAction, stdout: &str, stderr: &str) -> ReplCommandResult {
    ReplCommandResult {
        action,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_and_empty_input_match_c_responses() {
        let mut state = ReplState::new(ThinkMode::None, 128);

        let empty = state.handle_line("   ");
        assert_eq!(empty.action, ReplAction::Continue);
        assert!(empty.stdout.is_empty());
        assert!(empty.stderr.is_empty());

        let help = state.handle_line("/help");
        assert_eq!(help.action, ReplAction::Continue);
        assert!(help.stdout.contains("/think-max"));
        assert!(help.stdout.contains("Ctrl+C"));
        assert!(help.stderr.is_empty());
    }

    #[test]
    fn thinking_commands_update_state_and_messages() {
        let mut state = ReplState::new(ThinkMode::None, 128);

        let high = state.handle_line("/think");
        assert_eq!(state.think_mode, ThinkMode::High);
        assert_eq!(high.stdout, "Thinking mode: high.\n");
        assert!(high.stderr.is_empty());

        let max = state.handle_line("/think-max");
        assert_eq!(state.think_mode, ThinkMode::Max);
        assert_eq!(max.stdout, "Thinking mode: high (ctx below 393216).\n");
        assert!(max
            .stderr
            .contains("ds4: warning: /think-max needs --ctx >="));

        let none = state.handle_line("/nothink");
        assert_eq!(state.think_mode, ThinkMode::None);
        assert_eq!(none.stdout, "Thinking mode: none.\n");
    }

    #[test]
    fn ctx_command_returns_reset_action_and_validation_errors() {
        let mut state = ReplState::new(ThinkMode::Max, 128);

        let missing = state.handle_line("/ctx");
        assert_eq!(missing.action, ReplAction::Continue);
        assert_eq!(missing.stderr, "ds4: /ctx needs a positive integer\n");

        let invalid = state.handle_line("/ctx nope");
        assert_eq!(invalid.action, ReplAction::Exit(2));
        assert_eq!(invalid.stderr, "ds4: invalid value for /ctx: nope\n");

        let zero = state.handle_line("/ctx 0");
        assert_eq!(zero.action, ReplAction::Exit(2));
        assert_eq!(zero.stderr, "ds4: invalid value for /ctx: 0\n");

        let negative = state.handle_line("/ctx -1");
        assert_eq!(negative.action, ReplAction::Exit(2));
        assert_eq!(negative.stderr, "ds4: invalid value for /ctx: -1\n");

        let valid = state.handle_line("/ctx 256");
        assert_eq!(valid.action, ReplAction::SetContext(256));
        assert_eq!(state.ctx_size, 256);
        assert!(valid.stderr.contains("ds4: warning: /ctx needs --ctx >="));

        let max_ctx = state.handle_line("/ctx 393216");
        assert_eq!(max_ctx.action, ReplAction::SetContext(393216));
        assert_eq!(state.ctx_size, 393216);
        assert!(max_ctx.stderr.is_empty());
    }

    #[test]
    fn read_quit_unknown_prompt_and_interrupt_actions_match_c_categories() {
        let mut state = ReplState::new(ThinkMode::None, 128);

        let missing_read = state.handle_line("/read");
        assert_eq!(missing_read.stderr, "ds4: /read needs a file path\n");

        let read = state.handle_line("/read prompt.txt");
        assert_eq!(read.action, ReplAction::ReadFile("prompt.txt".to_string()));
        assert!(read.stdout.is_empty());
        assert!(read.stderr.is_empty());

        let unknown = state.handle_line("/definitely-unknown");
        assert_eq!(
            unknown.stderr,
            "ds4: unknown command: /definitely-unknown\nds4: type /help for commands\n"
        );

        let prompt = state.handle_line("Answer with one short noun: glacier.");
        assert_eq!(
            prompt.action,
            ReplAction::RunPrompt("Answer with one short noun: glacier.".to_string())
        );

        let interrupt = state.handle_interrupt_at_prompt();
        assert_eq!(interrupt.action, ReplAction::Continue);
        assert!(interrupt.stdout.is_empty());
        assert!(interrupt.stderr.is_empty());

        assert_eq!(state.handle_line("/quit").action, ReplAction::Quit);
        assert_eq!(state.handle_line("/exit").action, ReplAction::Quit);
    }

    #[test]
    fn command_prefix_matching_is_boundary_checked() {
        let mut state = ReplState::new(ThinkMode::None, 128);

        for command in ["/", "/c", "/ct", "/ctxx", "/rea", "/readx"] {
            let result = state.handle_line(command);
            assert_eq!(result.action, ReplAction::Continue);
            assert_eq!(
                result.stderr,
                format!("ds4: unknown command: {command}\nds4: type /help for commands\n")
            );
        }

        assert_eq!(
            state.handle_line("/ctx\t256").action,
            ReplAction::SetContext(256)
        );
        assert_eq!(
            state.handle_line("/read\tprompt.txt").action,
            ReplAction::ReadFile("prompt.txt".to_string())
        );
    }
}
