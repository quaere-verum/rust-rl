//! Worker-only selected representations. Full raw predictions remain in PreparedValuePoint.
use rl::{Result, core::action::Action, diagnostics::value::*};
#[derive(Debug, Clone, Copy)]
pub struct QuantileStatistics {
    pub mean: f64,
    pub standard_deviation: Option<f64>,
    pub lower_quantile: f64,
    pub lower_cvar: f64,
}

#[derive(Debug, Clone)]
pub enum PreparedValueProfile {
    Quantiles {
        cdf: Vec<[f64; 2]>,
        source_crossings: usize,
        statistics: QuantileStatistics,
    },
    Categorical {
        cdf: Vec<[f64; 2]>,
        statistics: QuantileStatistics,
    },
    Moments {
        summary: MomentSummary,
        utility: Option<f64>,
    },
}
impl PreparedValueProfile {
    pub(crate) fn cdf(&self) -> Option<&[[f64; 2]]> {
        match self {
            Self::Quantiles { cdf, .. } | Self::Categorical { cdf, .. } => Some(cdf),
            Self::Moments { .. } => None,
        }
    }
}
pub(crate) fn display_scalar(
    p: &ValuePrediction,
    action: Option<&Action>,
    output: &ValueOutput,
    utility: Option<&MomentUtility>,
) -> Result<f64> {
    if let ValuePrediction::Moments { .. } = p
        && *output == ValueOutput::StandardDeviation
    {
        // A missing cell keeps the input grid and every valid neighbor intact.
        return Ok(p.moment_summary()?.standard_deviation.unwrap_or(f64::NAN));
    }
    if matches!(p, ValuePrediction::Discrete { .. }) {
        let projected = if let Some(Action::Discrete(a)) = action {
            p.select_discrete(*a)?
        } else {
            p.greedy_discrete()?
        };
        projected.scalar(output, utility)
    } else {
        p.scalar(output, utility)
    }
}

pub(crate) fn selected_profile(
    p: &ValuePrediction,
    action: Option<&Action>,
    output: &ValueOutput,
    utility: Option<&MomentUtility>,
) -> Result<(f64, Option<PreparedValueProfile>)> {
    let distribution = match p {
        ValuePrediction::Quantiles { .. } => {
            let quantiles = p.ordered_quantiles()?;
            let crossings = quantiles.source_crossing_count();
            Some((quantiles.into_weighted_atoms()?, Some(crossings)))
        }
        ValuePrediction::CategoricalReturn { .. } => Some((p.weighted_atoms()?, None)),
        _ => None,
    };
    if let Some((q, crossings)) = distribution {
        let mean = q.mean();
        let variance = q.variance();
        let standard_deviation = variance.as_ref().ok().map(|value| value.sqrt());
        let value = match output {
            ValueOutput::ExpectedReturn => mean,
            ValueOutput::Quantile { probability } => q.quantile(*probability)?,
            ValueOutput::Cvar { probability } => q.lower_cvar(*probability)?,
            ValueOutput::Variance if crossings.is_none() => variance?,
            ValueOutput::StandardDeviation if crossings.is_none() => variance?.sqrt(),
            _ => p.scalar(output, utility)?,
        };
        let cdf = q.cdf_points()?;
        let statistics = QuantileStatistics {
            mean,
            standard_deviation,
            lower_quantile: q.quantile(0.1)?,
            lower_cvar: q.lower_cvar(0.1)?,
        };
        let profile = match crossings {
            Some(source_crossings) => PreparedValueProfile::Quantiles {
                cdf,
                statistics,
                source_crossings,
            },
            None => PreparedValueProfile::Categorical { cdf, statistics },
        };
        return Ok((value, Some(profile)));
    }
    let value = display_scalar(p, action, output, utility)?;
    let profile = if matches!(p, ValuePrediction::Moments { .. }) {
        Some(PreparedValueProfile::Moments {
            summary: p.moment_summary()?,
            utility: utility.and_then(|u| {
                if matches!(output, ValueOutput::Utility) {
                    Some(value)
                } else {
                    p.scalar(&ValueOutput::Utility, Some(u)).ok()
                }
            }),
        })
    } else {
        None
    };
    Ok((value, profile))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_fixed_support_profile_uses_predicted_masses() {
        let prediction = ValuePrediction::CategoricalReturn {
            atoms: vec![-5., -1., 3.],
            probabilities: vec![0.05, 0.2, 0.75],
        };
        let (output, profile) = selected_profile(
            &prediction,
            None,
            &ValueOutput::Cvar { probability: 0.1 },
            None,
        )
        .unwrap();
        assert!((output - (-3.)).abs() < 1e-12);
        let Some(PreparedValueProfile::Categorical { cdf, statistics }) = profile else {
            panic!("fixed support requires its weighted return profile");
        };
        assert_eq!(
            cdf,
            [
                [-5., 0.],
                [-5., 0.05],
                [-1., 0.05],
                [-1., 0.25],
                [3., 0.25],
                [3., 1.]
            ]
        );
        assert_eq!(statistics.lower_quantile, -1.);
        assert!((statistics.standard_deviation.unwrap() - 4.96_f64.sqrt()).abs() < 1e-12);
    }
    #[test]
    fn selected_profile_preserves_weighted_ordered_cdf_raw_crossing_and_statistic() {
        let p = ValuePrediction::Quantiles {
            raw: vec![4., -2., 1.],
            locations: vec![0.1, 0.4, 0.8],
            masses: vec![0.2, 0.3, 0.5],
        };
        let (value, profile) =
            selected_profile(&p, None, &ValueOutput::ExpectedReturn, None).unwrap();
        assert!((value - 1.9).abs() < 1e-12);
        let Some(PreparedValueProfile::Quantiles {
            cdf,
            source_crossings,
            statistics,
        }) = profile
        else {
            panic!()
        };
        assert_eq!(source_crossings, 1);
        assert!((statistics.mean - 1.9).abs() < 1e-12);
        assert!((statistics.standard_deviation.unwrap() - 5.49_f64.sqrt()).abs() < 1e-12);
        assert_eq!(statistics.lower_quantile, -2.);
        assert_eq!(statistics.lower_cvar, -2.);
        assert_eq!(
            cdf,
            vec![
                [-2., 0.],
                [-2., 0.2],
                [1., 0.2],
                [1., 0.5],
                [4., 0.5],
                [4., 1.]
            ]
        );
        assert!(matches!(p,ValuePrediction::Quantiles{raw,..}if raw==vec![4.,-2.,1.]));
    }
    #[test]
    fn selected_profile_signed_moments_and_actual_utility_have_no_distribution() {
        let p = ValuePrediction::Moments {
            mean: 2.,
            raw_second: 3.,
        };
        let (_, profile) = selected_profile(
            &p,
            None,
            &ValueOutput::ExpectedReturn,
            Some(&MomentUtility::MeanVariance { penalty: 2. }),
        )
        .unwrap();
        let Some(PreparedValueProfile::Moments { summary, utility }) = profile else {
            panic!()
        };
        assert_eq!(summary.variance, -1.);
        assert_eq!(summary.standard_deviation, None);
        assert_eq!(utility, Some(2.));
        assert!(matches!(
            selected_profile(
                &p,
                None,
                &ValueOutput::ExpectedReturn,
                Some(&MomentUtility::Unsupported {
                    name: "custom".into()
                })
            )
            .unwrap()
            .1,
            Some(PreparedValueProfile::Moments { utility: None, .. })
        ));
    }
    #[test]
    fn standard_deviation_renders_valid_rows_and_preserves_invalid_selected_moments() {
        let valid = ValuePrediction::Moments {
            mean: 0.,
            raw_second: 4.,
        };
        let invalid = ValuePrediction::Moments {
            mean: 2.,
            raw_second: 1.,
        };
        assert_eq!(
            display_scalar(&valid, None, &ValueOutput::StandardDeviation, None).unwrap(),
            2.
        );
        let (value, profile) =
            selected_profile(&invalid, None, &ValueOutput::StandardDeviation, None).unwrap();
        assert!(value.is_nan());
        assert!(
            matches!(profile, Some(PreparedValueProfile::Moments { summary, .. }) if summary.variance == -3.)
        );
    }

    #[test]
    fn selected_profile_full_head_admission() {
        for n in [4096, 4097] {
            let p = ValuePrediction::Quantiles {
                raw: vec![1.; n],
                locations: (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect(),
                masses: vec![1. / n as f64; n],
            };
            let r = selected_profile(&p, None, &ValueOutput::ExpectedReturn, None);
            if n == 4096 {
                assert!(
                    matches!(r.unwrap().1,Some(PreparedValueProfile::Quantiles{cdf,..})if cdf.len()==8192)
                );
            } else {
                assert!(r.is_err());
            }
        }
    }
}
