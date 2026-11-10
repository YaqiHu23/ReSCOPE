"""Download pinned upstream source and apply the paper-workflow iSTAR patch."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "BLEEP": ("https://github.com/bowang-lab/BLEEP.git", "23959670c7407510c1716a173184bdcc18833a35"),
    "istar": ("https://github.com/daviddaiweizhang/istar.git", "3cb0e5352a86df337f41c5841d0808da0003457b"),
}


def setup():
    (ROOT / "third_party").mkdir(exist_ok=True)
    for name, (url, revision) in REPOS.items():
        directory = ROOT / "third_party" / name
        if not directory.exists():
            # Root code only: BLEEP's example expression datasets are not needed.
            subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", url, str(directory)], check=True)
            subprocess.run(["git", "-C", str(directory), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"{directory} has a different revision; move it aside before setup")
        if name == "istar":
            patch = str(ROOT / "patches/istar-training.patch")
            command = ["git", "-C", str(directory), "apply"]
            if subprocess.run(command + ["--reverse", "--check", patch], capture_output=True).returncode == 0:
                continue
            subprocess.run(command + ["--check", patch], check=True)
            subprocess.run(command + [patch], check=True)
    print("Pinned BLEEP/iSTAR code ready. No model checkpoints were downloaded.")


if __name__ == "__main__":
    setup()
