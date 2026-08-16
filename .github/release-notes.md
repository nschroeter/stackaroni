---

## 📦 Downloads

| File | For |
|---|---|
| `stackaroni-@VERSION@-macos-aarch64.tar.gz` | macOS 15 or later, Apple Silicon |
| `stackaroni-@VERSION@-linux-x86_64.tar.gz` | Linux x86-64, glibc 2.39 or later (Ubuntu 24.04 and newer) |
| `stackaroni-@VERSION@-windows-x86_64.zip` | Windows 10/11, x86-64 |

Each archive holds the GUI, the `stackaroni-cli` headless runner, `README.md`,
`PARAMETERS.md` and the licence. `SHA256SUMS` covers all three.

## ▶️ Running it

**The builds are not code-signed.** There is no Apple Developer ID or Windows
certificate behind this, so both systems will object the first time.

**macOS** — the app is quarantined on download. Right-click `Stackaroni.app` → **Open**
→ **Open** confirms it once, after which it launches normally. Or from a terminal:

```sh
xattr -d com.apple.quarantine /Applications/Stackaroni.app
```

**Windows** — SmartScreen shows "Windows protected your PC". **More info** → **Run
anyway**.

**Linux** — the archive holds plain binaries; `chmod +x` if your unpacker dropped the
bit. Needs the usual windowing and GL runtime libraries, listed in the README.

Built for Apple Silicon only on macOS; Intel Macs need a build from source. The Linux
binary is linked against the runner's glibc and will not start on older distributions.

Input is 16-bit TIFF, already developed and denoised from RAW — see the README for what
is deliberately out of scope. `PARAMETERS.md` describes every exposed parameter and what
moving it costs.
