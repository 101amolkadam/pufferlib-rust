use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("puffer").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("PufferLib - High-performance RL in Rust"));
}

#[test]
fn test_cli_list() {
    let mut cmd = Command::cargo_bin("puffer").unwrap();
    cmd.arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("Available environments:"))
        .stdout(predicate::str::contains("cartpole"));
}

#[test]
fn test_cli_eval() {
    let mut cmd = Command::cargo_bin("puffer").unwrap();
    cmd.arg("eval")
        .arg("cartpole")
        .arg("--episodes")
        .arg("1") // Run just 1 episode for speed
        .assert()
        .success()
        .stdout(predicate::str::contains("Starting evaluation"));
}

#[test]
#[cfg(feature = "torch")]
fn test_cli_train_dry_run() {
    // Requires torch feature. If not enabled, this test might be skipped or fail gracefully if we cfg check it.
    // However, binary compilation usually enables default features.
    
    // We run a very short training session to verify the command works.
    let mut cmd = Command::cargo_bin("puffer").unwrap();
    cmd.arg("train")
        .arg("cartpole")
        .arg("--timesteps")
        .arg("10") // Very short run
        .arg("--num-envs")
        .arg("1")
        .assert()
        .success();
}
