// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The MLX fetch must not act on the repository the build was started from.
//!
//! `mlxcel-core`'s build script runs CMake, and FetchContent fetches MLX with
//! `git clone --no-checkout`, then `git checkout <pin> --` and
//! `git submodule update --recursive --init` inside the clone. `git clone`
//! ignores `GIT_DIR`; the other two obey it. Git exports `GIT_DIR` to every
//! hook, `git rebase -x` command and `git bisect run` command started from a
//! linked worktree, so a build reached through one of those ran the checkout
//! against the caller's repository and died with `fatal: unable to read tree
//! (<pin>)`, the pinned MLX commit being an object the caller does not have.
//!
//! `mlx_pin::remove_git_discovery_env` is what the build script calls first to
//! prevent that. It changes the process environment, so the test that
//! exercises it has to own the process: this file deliberately holds exactly
//! ONE `#[test]`. Do not add a second test here, and do not move this one into
//! the crate's unit-test binary, where it would race the `git`-spawning tests
//! in `mlx_pin.rs`.

use std::path::Path;
use std::process::{Command, Output};

use mlxcel_mlx_pin::mlx_pin;

fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success())
}

/// Run `git` in `dir` with the environment this process has at that moment,
/// which is the point: nothing here clears or sets a variable on the command.
fn git(dir: &Path, args: &[&str]) -> Output {
    Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=mlxcel test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .expect("git should run")
}

fn git_ok(dir: &Path, args: &[&str]) -> String {
    let out = git(dir, args);
    assert!(
        out.status.success(),
        "git {args:?} failed in {}: {}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn commit_file(repo: &Path, name: &str, contents: &str, message: &str) -> String {
    std::fs::write(repo.join(name), contents).expect("write fixture file");
    git_ok(repo, &["add", name]);
    git_ok(repo, &["commit", "-q", "-m", message]);
    git_ok(repo, &["rev-parse", "HEAD"])
}

#[test]
fn after_the_cleanup_a_fetch_checks_out_the_pin_in_its_own_clone() {
    if !git_available() {
        eprintln!("skipping: git is not on PATH");
        return;
    }
    let dir = tempfile::TempDir::new().expect("create fixture dir");

    // The stand-in for MLX. The pin is deliberately not the branch tip, as
    // with the real one, so a clone that never ran the checkout is not at it.
    let upstream = dir.path().join("upstream");
    std::fs::create_dir(&upstream).unwrap();
    git_ok(&upstream, &["init", "-q"]);
    let pin = commit_file(&upstream, "pinned.txt", "pinned\n", "pinned");
    commit_file(&upstream, "later.txt", "later\n", "tip");

    // The repository the build is started from, with a linked worktree: the
    // layout in which git hands `GIT_DIR` to the commands it runs.
    let caller = dir.path().join("caller");
    std::fs::create_dir(&caller).unwrap();
    git_ok(&caller, &["init", "-q"]);
    commit_file(&caller, "app.txt", "app\n", "app");
    let lane = dir.path().join("lane");
    git_ok(
        &caller,
        &[
            "worktree",
            "add",
            "-q",
            "-b",
            "lane",
            lane.to_str().unwrap(),
        ],
    );
    let lane_git_dir = caller.join(".git/worktrees/lane");
    assert!(lane_git_dir.join("index").exists());

    // Every listed name goes, not only the ones git exports today.
    //
    // SAFETY: `set_var` and the function under test mutate the process-global
    // environment, which is unsound while another thread reads it. This is the
    // only test in this binary, and it starts no thread of its own. The same
    // holds for the second block below.
    unsafe {
        for name in mlx_pin::GIT_DISCOVERY_ENV {
            std::env::set_var(name, "set-by-the-caller");
        }
        mlx_pin::remove_git_discovery_env();
    }
    for name in mlx_pin::GIT_DISCOVERY_ENV {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} is still set after the cleanup"
        );
    }

    // What a pre-commit hook inherits in that worktree. `git rebase -x` and
    // `git bisect run` export the first of the two.
    unsafe {
        std::env::set_var("GIT_DIR", &lane_git_dir);
        std::env::set_var("GIT_INDEX_FILE", lane_git_dir.join("index"));
        mlx_pin::remove_git_discovery_env();
    }

    // The three commands FetchContent's generated clone script runs.
    let build = dir.path().join("build/_deps");
    std::fs::create_dir_all(&build).unwrap();
    git_ok(
        &build,
        &[
            "clone",
            "-q",
            "--no-checkout",
            "--config",
            "advice.detachedHead=false",
            upstream.to_str().unwrap(),
            "mlx-src",
        ],
    );
    let mlx_src = build.join("mlx-src");
    git_ok(&mlx_src, &["checkout", "-q", &pin, "--"]);
    git_ok(&mlx_src, &["submodule", "update", "--recursive", "--init"]);

    assert_eq!(
        mlx_pin::check_fetched_head(&mlx_src, &pin),
        mlx_pin::HeadCheck::Match
    );
    assert!(mlx_src.join("pinned.txt").exists());
    assert!(!mlx_src.join("later.txt").exists());
}
