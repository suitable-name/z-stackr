//! Export file names from a template: `{name}_{algo}_{date}` and friends.
//!
//! The template is user text (the settings file, the export dialog), and several
//! of its values come from places the user does not control (a project name, a
//! source file's stem), so the module has three jobs:
//!
//! 1. [`render`] expands the variables. An unknown `{placeholder}`, or a `{` with
//!    no closing `}`, is left exactly as typed so a typo stays visible; the
//!    dialog lists them through [`unknown_placeholders`].
//! 2. [`sanitize_filename`] turns the rendered text into one safe file-name stem:
//!    no path separators (so no `..` traversal), no characters Windows rejects, no
//!    reserved device names (`CON`, `NUL`, ...), no trailing dots or spaces. It runs
//!    over the whole rendered string, not per variable, so no substitution can
//!    slip past it.
//! 3. [`unique_path`] picks a path that does not exist yet (`name (2).png`) so an
//!    export never overwrites an earlier one.
//!
//! The date and time are passed in ([`SystemTime`]) so the output is testable.
//! Everything here is plain Rust; `tests/filename_template_test.rs` drives it
//! without a window.
//!
//! # Variables
//!
//! | Variable       | Value                                                        |
//! | -------------- | ------------------------------------------------------------ |
//! | `{name}`       | the project name, else the first source file's stem          |
//! | `{algo}`       | the pass that made the result (`Apex`, `Relief`, `AI`)         |
//! | `{date}`       | `2026-10-01` (UTC)                                           |
//! | `{time}`       | `14-05-09` (UTC; dashes, because `:` is not allowed on Windows) |
//! | `{timestamp}`  | Unix seconds                                                 |
//! | `{n}`          | the number of source frames                                  |
//! | `{w}`, `{h}`   | the image size in pixels (`0` when it is not known)          |

use std::path::{Path, PathBuf};

use web_time::SystemTime;

/// The template an empty one falls back to; also the settings default.
pub const DEFAULT_TEMPLATE: &str = "{name}_stacked";

/// Every variable name the template understands, in display order.
pub const VARIABLES: &[&str] = &["name", "algo", "date", "time", "timestamp", "n", "w", "h"];

/// The stem used when the rendered name is empty after sanitising.
pub const FALLBACK_STEM: &str = "export";

/// Longest stem kept, in characters: leaves room for the folder, a ` (99)` suffix
/// and the extension under Windows' classic 260-character path limit.
pub const MAX_STEM_CHARS: usize = 120;

/// Windows' reserved device names: reserved with or without an extension, in any
/// case.
const RESERVED_WINDOWS_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// What a template can refer to. Strings are used raw: [`sanitize_filename`] runs
/// over the rendered result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemplateContext {
    /// `{name}`: the project name, else the first source file's stem.
    pub name: String,
    /// `{algo}`: the pass that made the result; empty when unknown.
    pub algo: String,
    /// `{n}`: the number of source frames.
    pub frames: usize,
    /// `{w}`: image width in pixels, `0` when unknown.
    pub width: u32,
    /// `{h}`: image height in pixels, `0` when unknown.
    pub height: u32,
}

/// One lexical piece of a template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token<'a> {
    /// Plain text.
    Text(&'a str),
    /// `{name}`; holds the text between the braces.
    Placeholder(&'a str),
    /// A `{` with no closing `}`; holds the brace and everything after it.
    Unterminated(&'a str),
}

/// Split `template` into text, placeholders and a possible unterminated tail.
fn tokens(template: &str) -> Vec<Token<'_>> {
    let mut out = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        if open > 0 {
            out.push(Token::Text(&rest[..open]));
        }
        let from_brace = &rest[open..];
        let Some(close) = from_brace.find('}') else {
            out.push(Token::Unterminated(from_brace));
            return out;
        };
        out.push(Token::Placeholder(&from_brace[1..close]));
        rest = &from_brace[close + 1..];
    }
    if !rest.is_empty() {
        out.push(Token::Text(rest));
    }
    out
}

/// A broken-down UTC instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CivilTime {
    /// The calendar year.
    pub year: u64,
    /// The month, 1 to 12.
    pub month: u64,
    /// The day of the month, 1 to 31.
    pub day: u64,
    /// The hour, 0 to 23.
    pub hour: u64,
    /// The minute, 0 to 59.
    pub minute: u64,
    /// The second, 0 to 59.
    pub second: u64,
}

/// Seconds since the Unix epoch; an instant before the epoch counts as the epoch.
fn unix_seconds(now: SystemTime) -> u64 {
    now.duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// The UTC calendar date and time of `secs` seconds after the Unix epoch.
///
/// Proleptic Gregorian, from Howard Hinnant's public-domain `civil_from_days`
/// algorithm, rearranged to unsigned arithmetic (the epoch is the earliest
/// instant it accepts).
#[must_use]
pub fn civil_from_unix_seconds(secs: u64) -> CivilTime {
    let days = secs / 86_400;
    let time_of_day = secs % 86_400;

    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);

    CivilTime {
        year,
        month,
        day,
        hour: time_of_day / 3600,
        minute: (time_of_day % 3600) / 60,
        second: time_of_day % 60,
    }
}

/// The value of `{name}`, or `None` for a variable this module does not know.
fn substitute(name: &str, ctx: &TemplateContext, now: SystemTime) -> Option<String> {
    let secs = unix_seconds(now);
    let civil = civil_from_unix_seconds(secs);
    Some(match name {
        "name" => ctx.name.clone(),
        "algo" => ctx.algo.clone(),
        "date" => format!("{:04}-{:02}-{:02}", civil.year, civil.month, civil.day),
        "time" => format!("{:02}-{:02}-{:02}", civil.hour, civil.minute, civil.second),
        "timestamp" => secs.to_string(),
        "n" => ctx.frames.to_string(),
        "w" => ctx.width.to_string(),
        "h" => ctx.height.to_string(),
        _ => return None,
    })
}

/// Expand every `{variable}` of `template`. Unknown placeholders and an
/// unterminated `{` stay exactly as written.
#[must_use]
pub fn render(template: &str, ctx: &TemplateContext, now: SystemTime) -> String {
    let mut out = String::with_capacity(template.len());
    for token in tokens(template) {
        match token {
            Token::Text(text) | Token::Unterminated(text) => out.push_str(text),
            Token::Placeholder(name) => {
                out.push_str(&substitute(name, ctx, now).unwrap_or_else(|| format!("{{{name}}}")));
            }
        }
    }
    out
}

/// The placeholders of `template` that [`render`] leaves literal, as typed
/// (`{foo}`, or `{oops` for a brace that never closes), each once, in order of
/// first appearance.
#[must_use]
pub fn unknown_placeholders(template: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for token in tokens(template) {
        let text = match token {
            Token::Text(_) => continue,
            Token::Placeholder(name) if VARIABLES.contains(&name) => continue,
            Token::Placeholder(name) => format!("{{{name}}}"),
            Token::Unterminated(rest) => rest.to_owned(),
        };
        if !found.contains(&text) {
            found.push(text);
        }
    }
    found
}

/// `true` for the characters Windows rejects in a file name, and for control
/// characters. They become `_` rather than vanishing, so `A/B` and `AB` stay
/// different names.
const fn is_forbidden(c: char) -> bool {
    matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') || c.is_control()
}

/// One safe file-name stem (no extension) out of arbitrary rendered text.
///
/// * every path separator and every character Windows rejects becomes `_`, so the
///   result is a single path component (`..` cannot survive: trailing dots go);
/// * leading spaces and trailing dots and spaces are trimmed;
/// * a reserved device name (`CON`, `lpt1`, `NUL.txt`, ...) gets a `_` prefix;
/// * the result is at most [`MAX_STEM_CHARS`] characters;
/// * nothing left becomes [`FALLBACK_STEM`].
#[must_use]
pub fn sanitize_filename(raw: &str) -> String {
    let replaced: String = raw
        .chars()
        .map(|c| if is_forbidden(c) { '_' } else { c })
        .collect();
    let mut cleaned: String = replaced
        .trim_start_matches(' ')
        .trim_end_matches(['.', ' '])
        .chars()
        .take(MAX_STEM_CHARS)
        .collect();
    // Cutting at the limit can expose a new trailing dot or space.
    while cleaned.ends_with(['.', ' ']) {
        cleaned.pop();
    }
    if cleaned.is_empty() {
        return FALLBACK_STEM.to_owned();
    }
    let before_extension = cleaned.split('.').next().unwrap_or(&cleaned).trim_end();
    if RESERVED_WINDOWS_NAMES
        .iter()
        .any(|reserved| reserved.eq_ignore_ascii_case(before_extension))
    {
        cleaned.insert(0, '_');
    }
    cleaned
}

/// The template to use: `template`, or [`DEFAULT_TEMPLATE`] when it is blank.
fn effective_template(template: &str) -> &str {
    if template.trim().is_empty() {
        DEFAULT_TEMPLATE
    } else {
        template
    }
}

/// The finished file-name stem for `template` at `now`: rendered and sanitised.
#[must_use]
pub fn stem_at(template: &str, ctx: &TemplateContext, now: SystemTime) -> String {
    sanitize_filename(&render(effective_template(template), ctx, now))
}

/// [`stem_at`] for the current time.
#[must_use]
pub fn stem(template: &str, ctx: &TemplateContext) -> String {
    stem_at(template, ctx, SystemTime::now())
}

/// What the export dialog shows under the template field: the stem the file
/// would get right now. Add the extension for the full file name.
#[must_use]
pub fn preview(template: &str, ctx: &TemplateContext) -> String {
    stem(template, ctx)
}

/// A path inside `dir` for `stem` and `ext` (no leading dot) that does not exist
/// yet: `stem.ext`, else `stem (2).ext`, `stem (3).ext`, and so on. An empty
/// `ext` gives a name without a dot.
#[must_use]
pub fn unique_path(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let name = |suffix: &str| {
        if ext.is_empty() {
            format!("{stem}{suffix}")
        } else {
            format!("{stem}{suffix}.{ext}")
        }
    };
    let mut candidate = dir.join(name(""));
    let mut number = 2_u32;
    while candidate.exists() {
        candidate = dir.join(name(&format!(" ({number})")));
        number = number.saturating_add(1);
    }
    candidate
}

/// The path an export of `template` takes inside `dir`: the sanitised stem, the
/// extension `ext`, and a numbered suffix when the file exists already.
#[must_use]
pub fn resolve_path(dir: &Path, template: &str, ctx: &TemplateContext, ext: &str) -> PathBuf {
    unique_path(dir, &stem(template, ctx), ext)
}
