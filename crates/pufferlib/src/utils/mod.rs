//! Utility functions.

/// Set global random seed
pub fn set_seed(_seed: u64) {
    // Note: This only sets Rust's RNG seed
    // tch uses its own seed setting via tch::manual_seed
    #[cfg(feature = "torch")]
    tch::manual_seed(_seed as i64);
}

/// Abbreviate large numbers for display
pub fn abbreviate(num: u64) -> String {
    if num < 1_000 {
        format!("{}", num)
    } else if num < 1_000_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else if num < 1_000_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    }
}

/// Format duration in human-readable form
pub fn format_duration(seconds: f64) -> String {
    if seconds < 0.0 {
        return "0s".to_string();
    }

    let secs = seconds as u64;
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;

    if h > 0 {
        format!("{}h {}m {}s", h, m, s)
    } else if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{}s", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abbreviate() {
        assert_eq!(abbreviate(500), "500");
        assert_eq!(abbreviate(1500), "1.5K");
        assert_eq!(abbreviate(1_500_000), "1.5M");
        assert_eq!(abbreviate(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3661.0), "1h 1m 1s");
    }
}
