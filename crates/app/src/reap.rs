//! Removing temp files left by runs that are no longer running.
//!
//! Keeping a *failed* run's scratch is deliberate — it is the case worth inspecting —
//! but nothing ever removed it, so it accumulated. Measured on this machine on
//! 2026-08-10: 38 GB of stale scratch from long-dead runs, plus an orphaned result. A
//! free-space check alone would only have half-solved the problem, since forgotten
//! debris is exactly what eats the headroom the check is protecting.
//!
//! # Why the pid, and not the file's age
//!
//! Age cannot distinguish "abandoned an hour ago" from "still running, slowly". A
//! 100-frame run takes ~20 minutes today and a larger stack takes longer, so any
//! threshold safe enough to never delete live scratch is long enough to leave tens of
//! GB sitting around. A pid is exact: if the process that created the directory is
//! gone, nothing is using it, whatever its age. The failure mode is also the safe one —
//! an unparseable name or a recycled pid means the entry is *kept*.

use std::path::{Path, PathBuf};

/// What a sweep removed.
pub struct Reaped {
    pub entries: usize,
    pub bytes: u64,
}

/// Delete every `stackaroni-*` entry in `dir` whose owning process has exited.
///
/// Never fails the caller: this is housekeeping, and a temp directory that cannot be
/// read or an entry that cannot be removed is not a reason to refuse to start.
pub fn stale(dir: &Path) -> Reaped {
    let mut reaped = Reaped {
        entries: 0,
        bytes: 0,
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return reaped;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Anything we cannot attribute to a dead process is left alone.
        let Some(pid) = owner_pid(name) else {
            continue;
        };
        if alive(pid) {
            continue;
        }

        let size = size_of(&path);
        let removed = if path.is_dir() {
            std::fs::remove_dir_all(&path).is_ok()
        } else {
            std::fs::remove_file(&path).is_ok()
        };
        if removed {
            reaped.entries += 1;
            reaped.bytes += size;
        }
    }
    reaped
}

/// The pid encoded in a temp entry's name, if it is one of ours.
///
/// Two shapes are in use and they put the pid in *different* positions, which is the
/// whole reason this is a function with tests rather than a `split('-').last()`:
///
/// - `stackaroni-run-<pid>-<sequence>` — the app, pid second, and `<sequence>` is a
///   small integer. Reading the last field here would parse the sequence as a pid, and
///   sequence 1 is pid 1, which is always alive — so app scratch would never be reaped.
/// - `stackaroni-<stack>-<pid>` — the CLI, pid last, and `<stack>` is a folder name that
///   may itself contain hyphens.
///
/// Values outside the range a real process id can take are rejected here rather than
/// being passed on, because on POSIX they are not merely invalid — they are *wildcards*.
/// `kill(0, sig)` addresses the caller's entire process group and `kill(-1, sig)`
/// addresses every process the caller may signal, so either would come back "alive" and,
/// read the other way round, could have made this ask a question about the wrong thing
/// entirely. `u32::MAX` reaches the second case by wrapping to `-1` in `pid_t`.
fn owner_pid(name: &str) -> Option<u32> {
    let stem = name.strip_suffix(".tif").unwrap_or(name);
    let rest = stem.strip_prefix("stackaroni-")?;
    let pid: u32 = match rest.strip_prefix("run-") {
        Some(tail) => tail.split('-').next()?.parse().ok()?,
        None => rest.rsplit('-').next()?.parse().ok()?,
    };
    (pid > 0 && pid <= i32::MAX as u32).then_some(pid)
}

fn size_of(path: &Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    if meta.is_file() {
        return meta.len();
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| size_of(&e.path()))
        .sum::<u64>()
        .to_owned()
}

/// Is a process with this id running?
///
/// Answering "yes" wrongly only leaves a stale directory in place; answering "no"
/// wrongly would delete a live run's scratch. Both implementations therefore lean
/// towards "alive" on anything ambiguous.
#[cfg(unix)]
fn alive(pid: u32) -> bool {
    // Signal 0 performs the permission and existence checks without delivering
    // anything. `EPERM` means the process exists but belongs to someone else, which is
    // still alive; only `ESRCH` means gone.
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if result == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(windows)]
fn alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    // SAFETY: a null handle is the documented failure return, and the handle is closed
    // on every path that obtained one.
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            // Most often "no such process". It can also be a permissions failure, in
            // which case the process exists — and keeping the directory is the safe
            // outcome either way.
            return false;
        }
        let mut code: u32 = 0;
        let queried = GetExitCodeProcess(handle, &mut code) != 0;
        CloseHandle(handle);
        // A pid can be reused by a *finished* process whose handle is still open; the
        // exit code distinguishes that from one still running.
        !queried || code == STILL_ACTIVE as u32
    }
}

/// Where runs put their working files.
pub fn temp_root() -> PathBuf {
    std::env::temp_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_is_read_from_the_right_field_for_both_shapes() {
        // The app's shape: pid second. Reading the last field would give the sequence
        // number, and sequence 1 parses as pid 1 — always alive — so nothing the app
        // ever wrote would be reaped.
        assert_eq!(owner_pid("stackaroni-run-48612-1"), Some(48612));
        assert_eq!(owner_pid("stackaroni-run-48612-1.tif"), Some(48612));

        // The CLI's shape: pid last, and the stack name may contain hyphens.
        assert_eq!(owner_pid("stackaroni-blossom-30799"), Some(30799));
        assert_eq!(owner_pid("stackaroni-my-odd-stack-1234"), Some(1234));
    }

    #[test]
    fn anything_unrecognised_is_left_alone() {
        // Deleting something we cannot attribute would be far worse than keeping it.
        assert_eq!(owner_pid("stackaroni-blossom-notapid"), None);
        assert_eq!(owner_pid("some-other-tool-1234"), None);
        assert_eq!(owner_pid("stackaroni-"), None);
        assert_eq!(owner_pid(".DS_Store"), None);
    }

    #[test]
    fn wildcard_pids_are_not_treated_as_process_ids() {
        // Found by a test rather than by reading: `kill(0, 0)` succeeds because it
        // addresses the caller's process group, so pid 0 read as "alive" and its files
        // were never reaped. `u32::MAX` is worse — it wraps to -1 in `pid_t`, which
        // addresses every signallable process.
        assert_eq!(owner_pid("stackaroni-gone-0"), None);
        assert_eq!(owner_pid("stackaroni-run-0-1"), None);
        assert_eq!(owner_pid(&format!("stackaroni-gone-{}", u32::MAX)), None);
    }

    /// A pid that is certainly not running: spawn a child and wait for it to exit.
    ///
    /// Better than picking a large number and hoping, which is only probably dead and
    /// would make this test flaky on a busy machine.
    fn exited_pid() -> u32 {
        let mut child = if cfg!(windows) {
            std::process::Command::new("cmd")
                .args(["/C", "exit"])
                .spawn()
        } else {
            std::process::Command::new("true").spawn()
        }
        .expect("spawning a trivial child");
        let pid = child.id();
        child.wait().expect("waiting for the child");
        pid
    }

    #[test]
    fn a_live_process_keeps_its_files_and_a_dead_one_does_not() {
        let dir = tempfile::tempdir().unwrap();

        // Ours, and we are obviously running.
        let live = dir
            .path()
            .join(format!("stackaroni-run-{}-1", std::process::id()));
        std::fs::create_dir(&live).unwrap();
        std::fs::write(live.join("focus.f32"), [0u8; 512]).unwrap();

        let dead = dir.path().join(format!("stackaroni-gone-{}", exited_pid()));
        std::fs::create_dir(&dead).unwrap();
        std::fs::write(dead.join("weights.f32"), [0u8; 1024]).unwrap();

        // Not ours at all.
        let other = dir.path().join("unrelated-file.txt");
        std::fs::write(&other, b"keep me").unwrap();

        let reaped = stale(dir.path());

        assert!(live.exists(), "a running process must keep its scratch");
        assert!(other.exists(), "unrelated files must be untouched");
        assert!(!dead.exists(), "a dead process's scratch should be gone");
        assert_eq!(reaped.entries, 1);
        assert_eq!(reaped.bytes, 1024);
    }
}
