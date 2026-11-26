#[path = "support/generic_run.rs"]
mod generic_run;
fn main() -> rl::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "runs/diagnostic-generic".into());
    let mut extra = std::env::args().skip(2);
    let categorical = match extra.next().as_deref() {
        None => false,
        Some("--categorical-demo") => true,
        Some(_) => {
            return Err(rl::Error::Config(
                "unknown diagnostic_generic option".into(),
            ));
        }
    };
    if extra.next().is_some() {
        return Err(rl::Error::Config("unexpected extra argument".into()));
    }
    let output = std::path::Path::new(&path);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    if categorical {
        generic_run::record_categorical_demo(output)?;
    } else {
        generic_run::record(output)?;
    }
    println!(
        "{} saved at {path}; open with cargo run -p rl-viewer -- {path}",
        if categorical {
            "categorical-counterbalance-seed7301"
        } else {
            generic_run::RUN_ID
        }
    );
    Ok(())
}
