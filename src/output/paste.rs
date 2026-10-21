//! Paste-based text output
//!
//! Uses wl-copy to copy text to clipboard, then simulates a paste keystroke.
//! This works around non-US keyboard layout issues by avoiding direct typing.
//!
//! Requires:
//! - wl-copy installed (for clipboard access)
//! - wtype OR eitype OR ydotool installed (for keystroke simulation)
//!   - wtype: Wayland-native, no daemon needed (preferred)
//!   - eitype: EI protocol, works on GNOME/KDE/Sway with libei
//!   - ydotool: Works on X11/Wayland/TTY, requires ydotoold daemon
//!
//! macOS: pbcopy/pbpaste for the clipboard and a synthetic CGEvent keystroke
//! for the paste step — no external tools. Keys with no Mac keycode (e.g.
//! `insert` from the shipped `shift+insert` default) fall back to cmd+v.
//! Needs the same Accessibility permission as type mode.

use super::TextOutput;
use crate::config::OutputDriver;
use crate::error::OutputError;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

#[cfg(not(target_os = "macos"))]
use super::session::{detect, DisplaySession};
#[cfg(not(target_os = "macos"))]
use crate::output::find_ydotool_socket;
#[cfg(not(target_os = "macos"))]
use crate::output::xclip::copy_to_x11_clipboard;

#[cfg(target_os = "macos")]
use super::pbcopy::PbcopyOutput;

/// Keystroke-capable drivers, in the order tried when the configured
/// `driver_order` doesn't name any of them. dotool/clipboard/xclip have no
/// keystroke-simulation path, so they never appear here.
const DEFAULT_KEYSTROKE_DRIVER_ORDER: [OutputDriver; 3] = [
    OutputDriver::Wtype,
    OutputDriver::Eitype,
    OutputDriver::Ydotool,
];

/// Filter a configured `[output] driver_order` down to keystroke-capable
/// drivers, preserving relative order. Falls back to
/// `DEFAULT_KEYSTROKE_DRIVER_ORDER` when the configured order is absent or
/// names none of wtype/eitype/ydotool (e.g. `["dotool", "clipboard"]`),
/// since paste mode's keystroke step has nothing to run in that case.
fn resolve_keystroke_driver_order(configured: Option<&[OutputDriver]>) -> Vec<OutputDriver> {
    let filtered: Vec<OutputDriver> = configured
        .map(|order| {
            order
                .iter()
                .copied()
                .filter(|d| DEFAULT_KEYSTROKE_DRIVER_ORDER.contains(d))
                .collect()
        })
        .unwrap_or_default();

    if filtered.is_empty() {
        DEFAULT_KEYSTROKE_DRIVER_ORDER.to_vec()
    } else {
        filtered
    }
}

/// Parsed paste keystroke (modifiers + key)
#[derive(Debug, Clone)]
struct ParsedKeystroke {
    /// Modifier keys (e.g., ["ctrl"], ["shift"], ["ctrl", "shift"])
    modifiers: Vec<String>,
    /// The main key (e.g., "v", "insert")
    key: String,
}

impl ParsedKeystroke {
    /// Parse a keystroke string like "ctrl+v" or "shift+insert"
    fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('+').map(|p| p.trim()).collect();

        if parts.is_empty() || parts.iter().any(|p| p.is_empty()) {
            return Err("Invalid keystroke format".to_string());
        }

        if parts.len() == 1 {
            // Just a key, no modifiers
            return Ok(Self {
                modifiers: vec![],
                key: parts[0].to_lowercase(),
            });
        }

        // Last part is the key, rest are modifiers
        let key = parts.last().unwrap().to_lowercase();
        let modifiers: Vec<String> = parts[..parts.len() - 1]
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        Ok(Self { modifiers, key })
    }

    /// Convert to wtype arguments
    /// e.g., "ctrl+v" -> ["-M", "ctrl", "-k", "v", "-m", "ctrl"]
    #[cfg(not(target_os = "macos"))]
    fn to_wtype_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Press modifiers
        for modifier in &self.modifiers {
            args.push("-M".to_string());
            args.push(modifier.clone());
        }

        // Tap the key
        args.push("-k".to_string());
        args.push(self.key.clone());

        // Release modifiers (reverse order)
        for modifier in self.modifiers.iter().rev() {
            args.push("-m".to_string());
            args.push(modifier.clone());
        }

        args
    }

    /// Convert to eitype arguments
    /// e.g., "ctrl+v" -> ["-M", "ctrl", "-k", "v"]
    #[cfg(not(target_os = "macos"))]
    fn to_eitype_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Press modifiers
        for modifier in &self.modifiers {
            args.push("-M".to_string());
            args.push(modifier.clone());
        }

        // Tap the key
        args.push("-k".to_string());
        args.push(self.key.clone());

        args
    }

    /// Convert to ydotool key arguments using evdev codes
    /// e.g., "ctrl+v" -> ["29:1", "47:1", "47:0", "29:0"]
    /// Convert to ydotool arguments
    #[cfg(not(target_os = "macos"))]
    fn to_ydotool_args(&self) -> Result<Vec<String>, String> {
        let mut args = Vec::new();

        // Get evdev codes for modifiers
        let modifier_codes: Vec<u16> = self
            .modifiers
            .iter()
            .map(|m| key_name_to_evdev(m))
            .collect::<Result<Vec<_>, _>>()?;

        // Get evdev code for main key
        let key_code = key_name_to_evdev(&self.key)?;

        // Press modifiers
        for code in &modifier_codes {
            args.push(format!("{}:1", code));
        }

        // Press and release the key
        args.push(format!("{}:1", key_code));
        args.push(format!("{}:0", key_code));

        // Release modifiers (reverse order)
        for code in modifier_codes.iter().rev() {
            args.push(format!("{}:0", code));
        }

        Ok(args)
    }
}

/// Convert a key name to its evdev code
#[cfg(not(target_os = "macos"))]
fn key_name_to_evdev(name: &str) -> Result<u16, String> {
    match name.to_lowercase().as_str() {
        // Modifiers
        "ctrl" | "control" | "leftctrl" => Ok(29), // KEY_LEFTCTRL
        "rightctrl" => Ok(97),                     // KEY_RIGHTCTRL
        "shift" | "leftshift" => Ok(42),           // KEY_LEFTSHIFT
        "rightshift" => Ok(54),                    // KEY_RIGHTSHIFT
        "alt" | "leftalt" => Ok(56),               // KEY_LEFTALT
        "rightalt" | "altgr" => Ok(100),           // KEY_RIGHTALT
        "super" | "meta" | "leftmeta" | "win" => Ok(125), // KEY_LEFTMETA

        // Common keys
        "v" => Ok(47),                // KEY_V
        "insert" | "ins" => Ok(110),  // KEY_INSERT
        "enter" | "return" => Ok(28), // KEY_ENTER

        // Letters (for completeness)
        "a" => Ok(30),
        "b" => Ok(48),
        "c" => Ok(46),
        "d" => Ok(32),
        "e" => Ok(18),
        "f" => Ok(33),
        "g" => Ok(34),
        "h" => Ok(35),
        "i" => Ok(23),
        "j" => Ok(36),
        "k" => Ok(37),
        "l" => Ok(38),
        "m" => Ok(50),
        "n" => Ok(49),
        "o" => Ok(24),
        "p" => Ok(25),
        "q" => Ok(16),
        "r" => Ok(19),
        "s" => Ok(31),
        "t" => Ok(20),
        "u" => Ok(22),
        "w" => Ok(17),
        "x" => Ok(45),
        "y" => Ok(21),
        "z" => Ok(44),

        other => Err(format!("Unknown key: {}", other)),
    }
}

/// Clipboard content with MIME type for restoration
#[derive(Clone)]
struct ClipboardContent {
    data: Vec<u8>,
    mime_type: String,
}

impl std::fmt::Debug for ClipboardContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipboardContent")
            .field("mime_type", &self.mime_type)
            .field("data", &format!("[{} bytes]", self.data.len()))
            .finish()
    }
}

/// Paste-based text output (clipboard + paste keystroke)
pub struct PasteOutput {
    /// Whether to send Enter key after output
    auto_submit: bool,
    /// Text to append after transcription (before auto_submit)
    append_text: Option<String>,
    /// Parsed paste keystroke
    keystroke: ParsedKeystroke,
    /// Delay between key events in milliseconds
    type_delay_ms: u32,
    /// Delay before pasting (after clipboard copy) in milliseconds
    pre_type_delay_ms: u32,
    /// Whether to restore clipboard content after paste
    restore_clipboard: bool,
    /// Delay after paste before restoring clipboard (milliseconds)
    restore_clipboard_delay_ms: u32,
    /// Order in which keystroke-capable drivers (wtype/eitype/ydotool) are
    /// tried for the paste keystroke and the auto-submit Enter. Derived from
    /// `[output] driver_order`; see [`resolve_keystroke_driver_order`].
    driver_order: Vec<OutputDriver>,
}

impl PasteOutput {
    /// Create a new paste output
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        auto_submit: bool,
        append_text: Option<String>,
        paste_keys: Option<String>,
        driver_order: Option<Vec<OutputDriver>>,
        type_delay_ms: u32,
        pre_type_delay_ms: u32,
        restore_clipboard: bool,
        restore_clipboard_delay_ms: u32,
    ) -> Self {
        let keystroke_str = paste_keys.as_deref().unwrap_or("ctrl+v");
        let keystroke = ParsedKeystroke::parse(keystroke_str).unwrap_or_else(|e| {
            tracing::warn!(
                "Invalid paste_keys '{}': {}, using ctrl+v",
                keystroke_str,
                e
            );
            ParsedKeystroke::parse("ctrl+v").unwrap()
        });

        tracing::debug!("Paste keystroke configured: {:?}", keystroke);

        let driver_order = resolve_keystroke_driver_order(driver_order.as_deref());
        tracing::debug!("Paste keystroke driver order: {:?}", driver_order);

        Self {
            auto_submit,
            append_text,
            keystroke,
            type_delay_ms,
            pre_type_delay_ms,
            restore_clipboard,
            restore_clipboard_delay_ms,
            driver_order,
        }
    }

    /// Copy text to clipboard, dispatching by platform/session.
    ///
    /// macOS uses pbcopy. Wayland sessions use `wl-copy`; X11 sessions use
    /// `xclip` (preferred) or `xsel` (fallback). Without this dispatch, X11
    /// users see voxtype silently no-op on the clipboard (GitHub #346).
    async fn copy_to_clipboard(&self, text: &str) -> Result<(), OutputError> {
        #[cfg(target_os = "macos")]
        {
            return self.copy_to_clipboard_macos(text).await;
        }

        #[cfg(not(target_os = "macos"))]
        self.copy_to_clipboard_unix(text).await
    }

    #[cfg(not(target_os = "macos"))]
    async fn copy_to_clipboard_unix(&self, text: &str) -> Result<(), OutputError> {
        if detect() == DisplaySession::X11 {
            return copy_to_x11_clipboard(text.as_bytes()).await;
        }

        // Spawn wl-copy with stdin pipe
        let mut child = Command::new("wl-copy")
            .arg("--type")
            .arg(super::clipboard::WL_COPY_TEXT_MIME)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::WlCopyNotFound
                } else {
                    OutputError::InjectionFailed(e.to_string())
                }
            })?;

        // Write text to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

            // Close stdin to signal EOF
            drop(stdin);
        }

        // Wait for completion
        let status = child
            .wait()
            .await
            .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

        if !status.success() {
            return Err(OutputError::InjectionFailed(
                "wl-copy exited with error".to_string(),
            ));
        }

        Ok(())
    }

    /// Read current clipboard content for restoration.
    async fn read_clipboard(&self) -> Result<Option<ClipboardContent>, OutputError> {
        #[cfg(target_os = "macos")]
        {
            return self.read_clipboard_macos().await;
        }

        #[cfg(not(target_os = "macos"))]
        self.read_clipboard_unix().await
    }

    /// Read clipboard using wl-paste (Wayland) or xclip (X11 fallback)
    #[cfg(not(target_os = "macos"))]
    async fn read_clipboard_unix(&self) -> Result<Option<ClipboardContent>, OutputError> {
        // Try wl-paste first (Wayland)
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            match self.read_clipboard_wl_paste().await {
                Ok(content) => return Ok(content),
                Err(e) => {
                    tracing::debug!("wl-paste failed, trying xclip: {}", e);
                }
            }
        }

        // Fallback to xclip (X11)
        self.read_clipboard_xclip().await
    }

    /// Read clipboard using wl-paste
    #[cfg(not(target_os = "macos"))]
    async fn read_clipboard_wl_paste(&self) -> Result<Option<ClipboardContent>, OutputError> {
        // First, check if clipboard is empty by listing types
        let types_output = Command::new("wl-paste")
            .arg("--list-types")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::WlPasteNotFound
                } else {
                    OutputError::InjectionFailed(e.to_string())
                }
            })?;

        if !types_output.status.success() {
            // Clipboard might be empty or error occurred
            let stderr = String::from_utf8_lossy(&types_output.stderr);
            tracing::debug!("wl-paste --list-types failed: {}", stderr);
            return Ok(None);
        }

        let types_str = String::from_utf8_lossy(&types_output.stdout);
        let mime_type = types_str
            .lines()
            .next()
            .unwrap_or("text/plain")
            .trim()
            .to_string();

        if mime_type.is_empty() {
            return Ok(None);
        }

        // Read the actual content
        let content_output = Command::new("wl-paste")
            .arg("--type")
            .arg(&mime_type)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

        if !content_output.status.success() {
            let stderr = String::from_utf8_lossy(&content_output.stderr);
            tracing::debug!("wl-paste failed to read content: {}", stderr);
            return Ok(None);
        }

        const MAX_CLIPBOARD_SIZE: usize = 100 * 1024 * 1024; // 100 MB
        if content_output.stdout.len() > MAX_CLIPBOARD_SIZE {
            tracing::warn!(
                "Clipboard content too large ({} bytes), skipping restoration",
                content_output.stdout.len()
            );
            return Ok(None);
        }

        Ok(Some(ClipboardContent {
            data: content_output.stdout,
            mime_type,
        }))
    }

    /// Read clipboard using xclip (X11 fallback)
    #[cfg(not(target_os = "macos"))]
    async fn read_clipboard_xclip(&self) -> Result<Option<ClipboardContent>, OutputError> {
        // Check if DISPLAY is set (X11 environment)
        if std::env::var("DISPLAY").is_err() {
            return Ok(None);
        }

        let output = Command::new("xclip")
            .args(["-selection", "clipboard", "-o"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::XclipNotFound
                } else {
                    OutputError::InjectionFailed(e.to_string())
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::debug!("xclip failed: {}", stderr);
            return Ok(None);
        }

        const MAX_CLIPBOARD_SIZE: usize = 100 * 1024 * 1024; // 100 MB
        if output.stdout.len() > MAX_CLIPBOARD_SIZE {
            tracing::warn!(
                "Clipboard content too large ({} bytes), skipping restoration",
                output.stdout.len()
            );
            return Ok(None);
        }

        // xclip doesn't provide MIME type, assume text/plain or infer from content
        let mime_type = if output.stdout.is_empty() {
            return Ok(None);
        } else {
            // Try to detect if it's text or binary
            match std::str::from_utf8(&output.stdout) {
                Ok(_) => "text/plain".to_string(),
                Err(_) => "application/octet-stream".to_string(),
            }
        };

        Ok(Some(ClipboardContent {
            data: output.stdout,
            mime_type,
        }))
    }

    /// Restore clipboard content using wl-copy, xclip, or pbcopy
    async fn restore_clipboard_content(
        &self,
        content: &ClipboardContent,
    ) -> Result<(), OutputError> {
        #[cfg(target_os = "macos")]
        {
            return self.restore_clipboard_content_macos(content).await;
        }

        #[cfg(not(target_os = "macos"))]
        self.restore_clipboard_content_unix(content).await
    }

    /// Restore clipboard content using wl-copy or xclip
    #[cfg(not(target_os = "macos"))]
    async fn restore_clipboard_content_unix(
        &self,
        content: &ClipboardContent,
    ) -> Result<(), OutputError> {
        // Try wl-copy first (Wayland)
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            match self.restore_clipboard_wl_copy(content).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::debug!("wl-copy restore failed, trying xclip: {}", e);
                }
            }
        }

        // Fallback to xclip
        self.restore_clipboard_xclip(content).await
    }

    /// Restore clipboard using wl-copy with MIME type preservation
    #[cfg(not(target_os = "macos"))]
    async fn restore_clipboard_wl_copy(
        &self,
        content: &ClipboardContent,
    ) -> Result<(), OutputError> {
        let mut child = Command::new("wl-copy")
            .arg("--type")
            .arg(&content.mime_type)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::WlCopyNotFound
                } else {
                    OutputError::InjectionFailed(e.to_string())
                }
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&content.data)
                .await
                .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;
            drop(stdin);
        }

        let status = child
            .wait()
            .await
            .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

        if !status.success() {
            return Err(OutputError::InjectionFailed(
                "wl-copy exited with error during restore".to_string(),
            ));
        }

        Ok(())
    }

    /// Restore clipboard using xclip
    #[cfg(not(target_os = "macos"))]
    async fn restore_clipboard_xclip(&self, content: &ClipboardContent) -> Result<(), OutputError> {
        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::XclipNotFound
                } else {
                    OutputError::InjectionFailed(e.to_string())
                }
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&content.data)
                .await
                .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;
            drop(stdin);
        }

        let status = child
            .wait()
            .await
            .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

        if !status.success() {
            return Err(OutputError::InjectionFailed(
                "xclip exited with error during restore".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if wtype is available
    #[cfg(not(target_os = "macos"))]
    async fn is_wtype_available(&self) -> bool {
        // Check if wtype exists
        let wtype_installed = Command::new("which")
            .arg("wtype")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);

        if !wtype_installed {
            return false;
        }

        // Check if we're on Wayland
        std::env::var("WAYLAND_DISPLAY").is_ok()
    }

    /// Check if eitype is available
    #[cfg(not(target_os = "macos"))]
    async fn is_eitype_available(&self) -> bool {
        Command::new("which")
            .arg("eitype")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Check if ydotool is available (installed and daemon running)
    #[cfg(not(target_os = "macos"))]
    async fn is_ydotool_available(&self) -> bool {
        // Check if ydotool exists
        let ydotool_installed = Command::new("which")
            .arg("ydotool")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);

        if !ydotool_installed {
            return false;
        }

        // Check if ydotoold is running by trying a no-op
        let mut cmd = Command::new("ydotool");
        if let Some(socket) = find_ydotool_socket() {
            cmd.env("YDOTOOL_SOCKET", socket);
        }
        cmd.args(["type", ""])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Simulate paste keystroke using wtype
    #[cfg(not(target_os = "macos"))]
    async fn simulate_paste_wtype(&self) -> Result<(), OutputError> {
        let args = self.keystroke.to_wtype_args();
        tracing::debug!("Running: wtype {}", args.join(" "));

        let output = Command::new("wtype")
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::WtypeNotFound
                } else {
                    OutputError::CtrlVFailed(e.to_string())
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(OutputError::CtrlVFailed(format!(
                "wtype failed: {}",
                stderr
            )));
        }

        Ok(())
    }

    /// Simulate paste keystroke using eitype
    #[cfg(not(target_os = "macos"))]
    async fn simulate_paste_eitype(&self) -> Result<(), OutputError> {
        let args = self.keystroke.to_eitype_args();
        tracing::debug!("Running: eitype {}", args.join(" "));

        let output = Command::new("eitype")
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::EitypeNotFound
                } else {
                    OutputError::CtrlVFailed(e.to_string())
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(OutputError::CtrlVFailed(format!(
                "eitype failed: {}",
                stderr
            )));
        }

        Ok(())
    }

    /// Simulate paste keystroke using ydotool
    #[cfg(not(target_os = "macos"))]
    async fn simulate_paste_ydotool(&self) -> Result<(), OutputError> {
        let args = self.keystroke.to_ydotool_args().map_err(|e| {
            OutputError::CtrlVFailed(format!("Cannot convert keystroke for ydotool: {}", e))
        })?;

        tracing::debug!(
            "Running: ydotool key {}, {}ms",
            args.join(" "),
            self.type_delay_ms
        );

        let mut cmd = Command::new("ydotool");
        if let Some(socket) = find_ydotool_socket() {
            cmd.env("YDOTOOL_SOCKET", socket);
        }
        cmd.arg("key");

        // Only add delay parameter if configured
        if self.type_delay_ms > 0 {
            cmd.arg("-d").arg(self.type_delay_ms.to_string());
        }

        let output = cmd
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::YdotoolNotFound
                } else {
                    OutputError::CtrlVFailed(e.to_string())
                }
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Check for common errors
            if stderr.contains("socket") || stderr.contains("connect") || stderr.contains("daemon")
            {
                return Err(OutputError::YdotoolNotRunning);
            }

            return Err(OutputError::CtrlVFailed(stderr.to_string()));
        }

        Ok(())
    }

    /// Simulate paste keystroke, trying each driver in `self.driver_order`
    /// (derived from `[output] driver_order`, defaulting to
    /// wtype -> eitype -> ydotool) until one succeeds.
    /// Simulate the paste keystroke.
    async fn simulate_paste_keystroke(&self) -> Result<(), OutputError> {
        #[cfg(target_os = "macos")]
        {
            return self.simulate_paste_keystroke_macos().await;
        }

        #[cfg(not(target_os = "macos"))]
        self.simulate_paste_keystroke_unix().await
    }

    /// Simulate paste keystroke using wtype/eitype/ydotool (Linux)
    #[cfg(not(target_os = "macos"))]
    async fn simulate_paste_keystroke_unix(&self) -> Result<(), OutputError> {
        let mut tried = Vec::new();

        for driver in &self.driver_order {
            let (available, name) = match driver {
                OutputDriver::Wtype => (self.is_wtype_available().await, "wtype"),
                OutputDriver::Eitype => (self.is_eitype_available().await, "eitype"),
                OutputDriver::Ydotool => (self.is_ydotool_available().await, "ydotool"),
                // driver_order is pre-filtered to the three keystroke-capable
                // drivers above; nothing else can reach this loop.
                _ => continue,
            };

            if !available {
                continue;
            }
            tried.push(name);

            let result = match driver {
                OutputDriver::Wtype => self.simulate_paste_wtype().await,
                OutputDriver::Eitype => self.simulate_paste_eitype().await,
                OutputDriver::Ydotool => self.simulate_paste_ydotool().await,
                _ => unreachable!(),
            };

            match result {
                Ok(()) => {
                    tracing::debug!("Paste keystroke sent via {}", name);
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!("{} paste failed: {}", name, e);
                }
            }
        }

        Err(OutputError::CtrlVFailed(format!(
            "No keystroke tool available (tried {})",
            if tried.is_empty() {
                "none".to_string()
            } else {
                tried.join(", ")
            }
        )))
    }

    /// Send Enter key after paste, trying the same driver order as the paste
    /// keystroke itself.
    async fn send_enter(&self) -> Result<(), OutputError> {
        #[cfg(target_os = "macos")]
        {
            return self.send_enter_macos().await;
        }

        #[cfg(not(target_os = "macos"))]
        self.send_enter_unix().await
    }

    #[cfg(not(target_os = "macos"))]
    async fn send_enter_unix(&self) -> Result<(), OutputError> {
        for driver in &self.driver_order {
            let available = match driver {
                OutputDriver::Wtype => self.is_wtype_available().await,
                OutputDriver::Eitype => self.is_eitype_available().await,
                OutputDriver::Ydotool => self.is_ydotool_available().await,
                _ => continue,
            };
            if !available {
                continue;
            }

            let output = match driver {
                OutputDriver::Wtype => {
                    Command::new("wtype")
                        .args(["-k", "Return"])
                        .stdout(Stdio::null())
                        .stderr(Stdio::piped())
                        .output()
                        .await
                }
                OutputDriver::Eitype => {
                    Command::new("eitype")
                        .args(["-k", "return"])
                        .stdout(Stdio::null())
                        .stderr(Stdio::piped())
                        .output()
                        .await
                }
                OutputDriver::Ydotool => {
                    let mut cmd = Command::new("ydotool");
                    if let Some(socket) = find_ydotool_socket() {
                        cmd.env("YDOTOOL_SOCKET", socket);
                    }
                    cmd.args(["key", "28:1", "28:0"])
                        .stdout(Stdio::null())
                        .stderr(Stdio::piped())
                        .output()
                        .await
                }
                _ => continue,
            };

            if let Ok(out) = output {
                if out.status.success() {
                    return Ok(());
                }
            }
        }

        // Best effort - don't fail the whole operation for Enter
        tracing::warn!("Failed to send Enter key");
        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn copy_to_clipboard_macos(&self, text: &str) -> Result<(), OutputError> {
        PbcopyOutput::new(false).output(text).await
    }

    #[cfg(target_os = "macos")]
    async fn read_clipboard_macos(&self) -> Result<Option<ClipboardContent>, OutputError> {
        let output = Command::new("pbpaste")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
            .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

        // pbpaste exits 0 with empty output both for an empty clipboard and
        // for text-free content; treat both as nothing to restore.
        if !output.status.success() || output.stdout.is_empty() {
            return Ok(None);
        }

        Ok(Some(ClipboardContent {
            data: output.stdout,
            mime_type: "text/plain".to_string(),
        }))
    }

    #[cfg(target_os = "macos")]
    async fn restore_clipboard_content_macos(
        &self,
        content: &ClipboardContent,
    ) -> Result<(), OutputError> {
        // pbcopy only carries UTF-8 text; non-text clipboard content can't be
        // restored on macOS, so keep the transcription instead.
        let text = match std::str::from_utf8(&content.data) {
            Ok(text) => text,
            Err(_) => {
                tracing::warn!("Clipboard restore skipped: non-UTF-8 content on macOS");
                return Ok(());
            }
        };

        let mut child = Command::new("pbcopy")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OutputError::InjectionFailed("pbcopy not found".to_string())
                } else {
                    OutputError::InjectionFailed(e.to_string())
                }
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;
            drop(stdin);
        }

        let status = child
            .wait()
            .await
            .map_err(|e| OutputError::InjectionFailed(e.to_string()))?;

        if !status.success() {
            return Err(OutputError::InjectionFailed(
                "pbcopy exited with error during restore".to_string(),
            ));
        }

        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn simulate_paste_keystroke_macos(&self) -> Result<(), OutputError> {
        // driver_order/type_delay_ms only shape the Linux keystroke drivers;
        // macOS always uses the CGEvent path.
        tracing::debug!(
            "macOS paste keystroke (type_delay_ms {} and driver_order {:?} ignored)",
            self.type_delay_ms,
            self.driver_order
        );
        let keystroke = self.keystroke.clone();
        tokio::task::spawn_blocking(move || macos_paste::send_keystroke(&keystroke))
            .await
            .map_err(|e| OutputError::InjectionFailed(format!("Task join error: {}", e)))?
    }

    #[cfg(target_os = "macos")]
    async fn send_enter_macos(&self) -> Result<(), OutputError> {
        tokio::task::spawn_blocking(macos_paste::post_return)
            .await
            .map_err(|e| OutputError::InjectionFailed(format!("Task join error: {}", e)))?
    }

    /// macOS probe: pbcopy/pbpaste are the only external requirements. The
    /// keystroke step uses CGEvent directly; a missing Accessibility
    /// permission surfaces as an actionable error at output time (same
    /// philosophy as the cgevent type-mode output, whose is_available always
    /// returns true to allow that helpful error).
    #[cfg(target_os = "macos")]
    async fn is_available_macos(&self) -> bool {
        for tool in ["pbcopy", "pbpaste"] {
            let installed = Command::new("which")
                .arg(tool)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await
                .map(|s| s.success())
                .unwrap_or(false);
            if !installed {
                tracing::debug!("paste mode unavailable: {} not found", tool);
                return false;
            }
        }
        true
    }
}

#[async_trait::async_trait]
impl TextOutput for PasteOutput {
    async fn output(&self, text: &str) -> Result<(), OutputError> {
        if text.is_empty() {
            return Ok(());
        }

        // Save original clipboard content if restoration is enabled
        let original_clipboard = if self.restore_clipboard {
            match self.read_clipboard().await {
                Ok(content) => {
                    if content.is_some() {
                        tracing::debug!("Saved clipboard content for restoration");
                    } else {
                        tracing::debug!("Clipboard was empty, nothing to restore");
                    }
                    content
                }
                Err(e) => {
                    tracing::warn!("Failed to read clipboard for restoration: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Prepare text with optional append
        let text_to_paste = if let Some(ref append) = self.append_text {
            format!("{}{}", text, append)
        } else {
            text.to_string()
        };

        // Step 1: Copy to clipboard
        self.copy_to_clipboard(&text_to_paste).await?;

        // Pre-paste delay to ensure clipboard is set before pasting
        // Default to 100ms if not configured (minimum needed for reliability)
        let delay = if self.pre_type_delay_ms > 0 {
            self.pre_type_delay_ms
        } else {
            100
        };
        tokio::time::sleep(std::time::Duration::from_millis(delay as u64)).await;

        // Step 2: Simulate paste keystroke
        self.simulate_paste_keystroke().await?;

        // Send Enter key if configured
        if self.auto_submit {
            self.send_enter().await?;
        }

        // Restore original clipboard content if we saved something
        if let Some(content) = original_clipboard {
            // Wait for paste to complete before restoring
            tokio::time::sleep(std::time::Duration::from_millis(
                self.restore_clipboard_delay_ms as u64,
            ))
            .await;

            match self.restore_clipboard_content(&content).await {
                Ok(()) => {
                    tracing::debug!("Restored original clipboard content");
                }
                Err(e) => {
                    tracing::warn!("Failed to restore clipboard content: {}", e);
                }
            }
        }

        tracing::info!(
            "Text pasted via clipboard + {} ({} chars)",
            self.keystroke.modifiers.join("+")
                + if !self.keystroke.modifiers.is_empty() {
                    "+"
                } else {
                    ""
                }
                + &self.keystroke.key,
            text.len()
        );
        Ok(())
    }

    async fn is_available(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            return self.is_available_macos().await;
        }

        #[cfg(not(target_os = "macos"))]
        self.is_available_unix().await
    }

    /// Probe the appropriate clipboard tool for the active session.
    /// Wayland needs wl-copy; X11 needs xclip or xsel.
    #[cfg(not(target_os = "macos"))]
    async fn is_available_unix(&self) -> bool {
        let session = detect();
        let clipboard_available = match session {
            DisplaySession::Wayland => Command::new("which")
                .arg("wl-copy")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .await
                .map(|s| s.success())
                .unwrap_or(false),
            DisplaySession::X11 => {
                let xclip_ok = Command::new("which")
                    .arg("xclip")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .await
                    .map(|s| s.success())
                    .unwrap_or(false);
                let xsel_ok = Command::new("which")
                    .arg("xsel")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .await
                    .map(|s| s.success())
                    .unwrap_or(false);
                xclip_ok || xsel_ok
            }
        };

        if !clipboard_available {
            tracing::debug!(
                "paste mode unavailable: no clipboard tool for {:?} session",
                session
            );
            return false;
        }

        // Check if wtype, eitype, or ydotool is available for keystroke simulation
        let wtype_available = self.is_wtype_available().await;
        let eitype_available = self.is_eitype_available().await;
        let ydotool_available = self.is_ydotool_available().await;

        if !wtype_available && !eitype_available && !ydotool_available {
            tracing::debug!(
                "paste mode unavailable: no keystroke tool available \
                (wtype needs WAYLAND_DISPLAY, eitype needs libei, ydotool needs daemon running)"
            );
            return false;
        }

        tracing::debug!(
            "paste mode available (wtype: {}, eitype: {}, ydotool: {})",
            wtype_available,
            eitype_available,
            ydotool_available
        );
        true
    }

    fn name(&self) -> &'static str {
        if cfg!(target_os = "macos") {
            "paste (pbcopy + CGEvent keystroke)"
        } else {
            "paste (clipboard + keystroke)"
        }
    }
}

#[cfg(target_os = "macos")]
mod macos_paste {
    //! macOS keystroke synthesis for paste mode.

    use super::{OutputError, ParsedKeystroke};
    use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

    /// Carbon HIToolbox virtual keycode for V.
    pub(super) const KEYCODE_V: CGKeyCode = 0x09;
    /// Carbon HIToolbox virtual keycode for Return.
    pub(super) const KEYCODE_RETURN: CGKeyCode = 0x24;

    /// Map a keystroke's key name to a Carbon virtual keycode.
    ///
    /// Only keys paste mode plausibly binds are covered. Notably `insert` is
    /// absent: Mac keyboards have no Insert key, so the shipped
    /// `shift+insert` default falls back to cmd+v.
    pub(super) fn keycode_for_key(key: &str) -> Option<CGKeyCode> {
        let code: CGKeyCode = match key {
            "a" => 0x00,
            "s" => 0x01,
            "d" => 0x02,
            "f" => 0x03,
            "h" => 0x04,
            "g" => 0x05,
            "z" => 0x06,
            "x" => 0x07,
            "c" => 0x08,
            "v" => 0x09,
            "b" => 0x0B,
            "q" => 0x0C,
            "w" => 0x0D,
            "e" => 0x0E,
            "r" => 0x0F,
            "y" => 0x10,
            "t" => 0x11,
            "1" => 0x12,
            "2" => 0x13,
            "3" => 0x14,
            "4" => 0x15,
            "6" => 0x16,
            "5" => 0x17,
            "9" => 0x19,
            "7" => 0x1A,
            "8" => 0x1B,
            "0" => 0x1D,
            "o" => 0x1F,
            "u" => 0x20,
            "i" => 0x22,
            "p" => 0x23,
            "l" => 0x25,
            "j" => 0x26,
            "k" => 0x28,
            "n" => 0x2D,
            "m" => 0x2E,
            "return" | "enter" => KEYCODE_RETURN,
            "tab" => 0x30,
            "space" => 0x31,
            "delete" | "backspace" => 0x33,
            "escape" | "esc" => 0x35,
            _ => return None,
        };
        Some(code)
    }

    /// Map a parsed paste keystroke to macOS modifier flags and a virtual
    /// keycode, falling back to cmd+v for keys with no Mac keycode.
    pub(super) fn keystroke_spec(keystroke: &ParsedKeystroke) -> (CGEventFlags, CGKeyCode) {
        let mut flags = CGEventFlags::empty();
        for modifier in &keystroke.modifiers {
            flags |= match modifier.as_str() {
                "ctrl" | "control" | "ctl" => CGEventFlags::CGEventFlagControl,
                "cmd" | "command" | "meta" | "super" | "win" => CGEventFlags::CGEventFlagCommand,
                "alt" | "option" | "opt" => CGEventFlags::CGEventFlagAlternate,
                "shift" => CGEventFlags::CGEventFlagShift,
                other => {
                    tracing::debug!("Unknown paste modifier '{}' ignored on macOS", other);
                    CGEventFlags::empty()
                }
            };
        }

        match keycode_for_key(&keystroke.key) {
            Some(code) => (flags, code),
            None => {
                tracing::warn!(
                    "paste key '{}' has no macOS keycode; falling back to cmd+v",
                    keystroke.key
                );
                (CGEventFlags::CGEventFlagCommand, KEYCODE_V)
            }
        }
    }

    /// Post one synthetic key event (down + up) with explicit flags, matching
    /// the cgevent type-mode output's approach to stuck modifiers.
    fn post_key(
        source: &CGEventSource,
        keycode: CGKeyCode,
        flags: CGEventFlags,
    ) -> Result<(), OutputError> {
        let down = CGEvent::new_keyboard_event(source.clone(), keycode, true)
            .map_err(|_| OutputError::InjectionFailed("Failed to create keyboard event".into()))?;
        down.set_flags(flags);
        down.post(CGEventTapLocation::HID);

        let up = CGEvent::new_keyboard_event(source.clone(), keycode, false)
            .map_err(|_| OutputError::InjectionFailed("Failed to create keyboard event".into()))?;
        up.set_flags(flags);
        up.post(CGEventTapLocation::HID);

        Ok(())
    }

    fn accessibility_error() -> OutputError {
        OutputError::InjectionFailed(
            "Accessibility permission required.\n\
             Grant access in: System Settings > Privacy & Security > Accessibility\n\
             Then restart voxtype."
                .into(),
        )
    }

    pub(super) fn send_keystroke(keystroke: &ParsedKeystroke) -> Result<(), OutputError> {
        if !crate::output::cgevent::CGEventOutput::check_accessibility_permission() {
            return Err(accessibility_error());
        }

        let (flags, keycode) = keystroke_spec(keystroke);
        tracing::debug!(
            "macOS paste keystroke: keycode {} with flags {:?}",
            keycode,
            flags
        );

        let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|_| OutputError::InjectionFailed("Failed to create CGEventSource".into()))?;

        post_key(&source, keycode, flags)
    }

    pub(super) fn post_return() -> Result<(), OutputError> {
        if !crate::output::cgevent::CGEventOutput::check_accessibility_permission() {
            return Err(accessibility_error());
        }

        let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|_| OutputError::InjectionFailed("Failed to create CGEventSource".into()))?;

        post_key(&source, KEYCODE_RETURN, CGEventFlags::empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stores_restore_clipboard_fields() {
        let output = PasteOutput::new(false, None, None, None, 10, 100, true, 300);
        assert!(output.restore_clipboard);
        assert_eq!(output.restore_clipboard_delay_ms, 300);
    }

    #[test]
    fn test_new_defaults_restore_clipboard_disabled() {
        let output = PasteOutput::new(false, None, None, None, 10, 100, false, 200);
        assert!(!output.restore_clipboard);
        assert_eq!(output.restore_clipboard_delay_ms, 200);
    }

    #[test]
    fn test_driver_order_defaults_to_wtype_eitype_ydotool() {
        let output = PasteOutput::new(false, None, None, None, 10, 100, false, 200);
        assert_eq!(
            output.driver_order,
            vec![
                OutputDriver::Wtype,
                OutputDriver::Eitype,
                OutputDriver::Ydotool
            ]
        );
    }

    #[test]
    fn test_driver_order_honours_config_override() {
        // #750: ydotool sends real evdev keycodes and survives XWayland/RDP
        // keycode remapping that breaks wtype; a user who puts it first in
        // driver_order must have paste mode actually try it first.
        let output = PasteOutput::new(
            false,
            None,
            None,
            Some(vec![OutputDriver::Ydotool, OutputDriver::Wtype]),
            10,
            100,
            false,
            200,
        );
        assert_eq!(
            output.driver_order,
            vec![OutputDriver::Ydotool, OutputDriver::Wtype]
        );
    }

    #[test]
    fn test_driver_order_falls_back_when_config_has_no_keystroke_driver() {
        // A driver_order of e.g. ["dotool", "clipboard"] (valid for type
        // mode) names nothing paste mode can use for the keystroke step, so
        // it must fall back to the default rather than trying nothing.
        let output = PasteOutput::new(
            false,
            None,
            None,
            Some(vec![OutputDriver::Dotool, OutputDriver::Clipboard]),
            10,
            100,
            false,
            200,
        );
        assert_eq!(
            output.driver_order,
            vec![
                OutputDriver::Wtype,
                OutputDriver::Eitype,
                OutputDriver::Ydotool
            ]
        );
    }

    #[test]
    fn test_clipboard_content_debug_redacts_data() {
        let content = ClipboardContent {
            data: vec![1, 2, 3, 4, 5],
            mime_type: "text/plain".to_string(),
        };
        let debug_str = format!("{:?}", content);
        assert!(debug_str.contains("[5 bytes]"));
        assert!(debug_str.contains("text/plain"));
        assert!(!debug_str.contains("[1, 2, 3"));
    }

    #[cfg(target_os = "macos")]
    mod macos_tests {
        use super::*;
        use core_graphics::event::CGEventFlags;

        #[test]
        fn test_keycode_mapping_covers_paste_keys() {
            assert_eq!(macos_paste::keycode_for_key("v"), Some(0x09));
            assert_eq!(macos_paste::keycode_for_key("c"), Some(0x08));
            assert_eq!(macos_paste::keycode_for_key("return"), Some(0x24));
            // Macs have no Insert key: the shipped shift+insert default must
            // not silently map to a wrong keycode.
            assert_eq!(macos_paste::keycode_for_key("insert"), None);
        }

        #[test]
        fn test_keystroke_spec_maps_ctrl_v() {
            let keystroke = ParsedKeystroke::parse("ctrl+v").unwrap();
            let (flags, code) = macos_paste::keystroke_spec(&keystroke);
            assert_eq!(code, 0x09);
            assert_eq!(flags, CGEventFlags::CGEventFlagControl);
        }

        #[test]
        fn test_keystroke_spec_maps_cmd_v() {
            let keystroke = ParsedKeystroke::parse("cmd+v").unwrap();
            let (flags, code) = macos_paste::keystroke_spec(&keystroke);
            assert_eq!(code, 0x09);
            assert_eq!(flags, CGEventFlags::CGEventFlagCommand);
        }

        #[test]
        fn test_keystroke_spec_falls_back_to_cmd_v_for_shift_insert() {
            // Shipped default has no macOS meaning; paste mode must still work.
            let keystroke = ParsedKeystroke::parse("shift+insert").unwrap();
            let (flags, code) = macos_paste::keystroke_spec(&keystroke);
            assert_eq!(code, macos_paste::KEYCODE_V);
            assert_eq!(flags, CGEventFlags::CGEventFlagCommand);
        }
    }
}
