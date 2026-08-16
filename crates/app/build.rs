//! Compiles the Windows icon into the executable.
//!
//! `ViewportBuilder::with_icon` covers the window and the taskbar, but the icon Explorer
//! draws for `stackaroni-app.exe` itself is a *resource* linked into the binary, and
//! nothing at runtime can set it. That is what this is for; on any other target it does
//! nothing at all.
//!
//! `.ico` rather than the `.png` the other platforms use, because that is the only format
//! a Windows icon resource takes. It is committed rather than generated here so the build
//! needs no image tooling — see `packaging/icon/README.md`.
//!
//! **Two different `windows` conditions appear below, and they are not the same one.**
//! `#[cfg(windows)]` is the *host*, and has to match the `cfg(windows)` gate on the
//! build-dependency in Cargo.toml — a build script is compiled for the machine running
//! it, so referring to a crate that gate excluded is a compile error rather than dead
//! code. `CARGO_CFG_TARGET_OS` is what is being *built for*. Both must hold: the resource
//! only belongs in a Windows binary, and only `rc.exe` from the Windows SDK can make it.

fn main() {
    #[cfg(windows)]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
            return;
        }

        // Cargo only re-runs this when told; without it a changed icon would not reach
        // an incremental build.
        println!("cargo:rerun-if-changed=../../packaging/icon/stackaroni.ico");

        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("../../packaging/icon/stackaroni.ico");
        // Loud rather than silent: an executable that builds without its icon looks fine
        // until it is in front of a user, and the cause is invisible by then.
        resource
            .compile()
            .expect("failed to compile the Windows icon resource");
    }
}
