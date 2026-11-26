//! Opt-in joint actor/value gradient clipping for independent PPO modules.
use std::collections::HashSet;

use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::GradientsParams,
    tensor::Tensor,
};

use super::ppo::PpoConfig;
use crate::{Result, core::ensure};

pub(super) fn valid_settings(
    config: &PpoConfig,
    value_loss_coefficient: f32,
    max_grad_norm: f32,
) -> bool {
    value_loss_coefficient.is_finite()
        && value_loss_coefficient >= 0.
        && max_grad_norm.is_finite()
        && max_grad_norm > 0.
        && config.policy.max_grad_norm.is_none()
        && config.value.max_grad_norm.is_none()
}

struct Norm<'a> {
    grads: &'a GradientsParams,
    visited: HashSet<ParamId>,
    squared: f64,
}
impl ModuleVisitor for Norm<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if self.visited.insert(param.id)
            && let Some(grad) = self.grads.get::<D>(param.id)
        {
            self.squared += grad.square().sum().into_scalar::<f32>() as f64;
        }
    }
}

struct Scale<'a> {
    grads: &'a mut GradientsParams,
    factor: f32,
    visited: HashSet<ParamId>,
}
impl ModuleVisitor for Scale<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if self.visited.insert(param.id)
            && let Some(grad) = self.grads.remove::<D>(param.id)
        {
            self.grads.register(param.id, grad * self.factor);
        }
    }
}

pub(super) fn clip<P: Module, V: Module>(
    policy: &P,
    value: &V,
    policy_grads: &mut GradientsParams,
    value_grads: &mut GradientsParams,
    limit: f32,
) -> Result<f32> {
    let mut p = Norm {
        grads: policy_grads,
        visited: HashSet::new(),
        squared: 0.,
    };
    policy.visit(&mut p);
    let mut v = Norm {
        grads: value_grads,
        visited: HashSet::new(),
        squared: 0.,
    };
    value.visit(&mut v);
    let norm = (p.squared + v.squared).sqrt();
    ensure(norm.is_finite(), "nonfinite joint PPO gradient norm")?;
    let factor = (limit as f64 / (norm + 1e-6)).min(1.) as f32;
    policy.visit(&mut Scale {
        grads: policy_grads,
        factor,
        visited: HashSet::new(),
    });
    value.visit(&mut Scale {
        grads: value_grads,
        factor,
        visited: HashSet::new(),
    });
    Ok(norm as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{nn::LinearConfig, tensor::TensorData};

    #[test]
    fn joint_clipping_handles_zero_and_missing_gradients() {
        let device = crate::nn::device();
        let policy = LinearConfig::new(1, 1).init(&device);
        let value = LinearConfig::new(1, 1).init(&device);
        let mut pg = GradientsParams::new();
        let mut vg = GradientsParams::new();
        pg.register(policy.weight.id, Tensor::<2>::from_data([[0.0]], &device));
        assert_eq!(clip(&policy, &value, &mut pg, &mut vg, 0.1).unwrap(), 0.);
        assert_eq!(
            pg.get::<2>(policy.weight.id).unwrap().into_scalar::<f32>(),
            0.
        );
        assert!(vg.is_empty());
        assert_eq!((0.1_f64 / (0.0_f64 + 1e-6)).min(1.0) as f32, 1.0);
    }

    #[test]
    fn captured_sb3_joint_gradient_reduction() {
        use crate::{
            core::action::ActionSpace,
            nn::{critics::scalar::Value, matrix, policy::Policy, to_vec},
        };
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let input: serde_json::Value = serde_json::from_slice(
            &std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/sb3_lqg_update105_gradients.json"
            ))
            .expect("generate captured SB3 gradient fixture"),
        )
        .unwrap();
        assert_eq!(input["update"], 105);
        assert_eq!(
            input["fixture_sha256"],
            "d782b0f84844f0649ab08f132afb50d271ad431fa2ca1caae3cd828283fba7b2"
        );
        let entries = input["gradients"].as_array().unwrap();
        assert_eq!(entries.len(), 13);
        let get = |name: &str| {
            let mut found = entries.iter().filter(|e| e["name"] == name);
            let entry = found.next().expect("missing named SB3 gradient");
            assert!(found.next().is_none(), "duplicate gradient name");
            entry
        };
        let as_f32 = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let policy = Policy::new(
            2,
            ActionSpace::Unbounded { dimensions: 1 },
            &[32, 32],
            &mut rng,
        )
        .unwrap();
        let value = Value::new(2, &[32, 32], &mut rng).unwrap();
        let mut pg = GradientsParams::new();
        let mut vg = GradientsParams::new();
        let actor_names = [
            "mlp_extractor.policy_net.0",
            "mlp_extractor.policy_net.2",
            "action_net",
        ];
        let value_names = [
            "mlp_extractor.value_net.0",
            "mlp_extractor.value_net.2",
            "value_net",
        ];
        for (network, names, grads) in [
            (&policy.network, actor_names, &mut pg),
            (&value.network, value_names, &mut vg),
        ] {
            for (layer, name) in network.layers.iter().zip(names) {
                let weight = get(&format!("{name}.weight"));
                let bias = get(&format!("{name}.bias"));
                let shape = weight["shape"].as_array().unwrap();
                let dims = [
                    shape[0].as_u64().unwrap() as usize,
                    shape[1].as_u64().unwrap() as usize,
                ];
                assert_eq!(layer.weight.val().dims(), dims);
                let weight_values = as_f32(&weight["before"]);
                let bias_values = as_f32(&bias["before"]);
                assert!(
                    weight_values
                        .iter()
                        .chain(&bias_values)
                        .all(|v| v.is_finite())
                );
                grads.register(layer.weight.id, matrix(weight_values, dims[0], dims[1]));
                grads.register(
                    layer.bias.as_ref().unwrap().id,
                    Tensor::<1>::from_data(
                        TensorData::new(bias_values, [dims[1]]),
                        &crate::nn::device(),
                    ),
                );
            }
        }
        let log_std = as_f32(&get("log_std")["before"]);
        assert_eq!(log_std.len(), 1);
        pg.register(
            policy.log_std.id,
            Tensor::<1>::from_data(TensorData::new(log_std, [1]), &crate::nn::device()),
        );
        let mut policy_norm = Norm {
            grads: &pg,
            visited: HashSet::new(),
            squared: 0.,
        };
        policy.visit(&mut policy_norm);
        let mut value_norm = Norm {
            grads: &vg,
            visited: HashSet::new(),
            squared: 0.,
        };
        value.visit(&mut value_norm);
        let norm = (policy_norm.squared + value_norm.squared).sqrt();
        let factor = (0.5_f64 / (norm + 1e-6)).min(1.) as f32;
        let reference_norm = input["pytorch_norm"].as_f64().unwrap();
        let reference_factor = input["pytorch_factor"].as_f64().unwrap();
        assert!(reference_norm.is_finite() && reference_factor.is_finite());
        assert!((norm - reference_norm).abs() <= 1e-4);
        assert!((factor as f64 - reference_factor).abs() <= 1e-7);
        let reported_norm = clip(&policy, &value, &mut pg, &mut vg, 0.5).unwrap();
        assert_eq!(reported_norm, norm as f32);
        let mut max_error = 0_f32;
        let mut max_relative = 0_f32;
        let mut count = 0_usize;
        let mut compare = |actual: Vec<f32>, expected: Vec<f32>| {
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.into_iter().zip(expected) {
                assert!(a.is_finite() && e.is_finite());
                let error = (a - e).abs();
                assert!(error <= 1e-9 + 1e-6 * e.abs(), "scaled gradient mismatch");
                max_error = max_error.max(error);
                if e != 0. {
                    max_relative = max_relative.max(error / e.abs());
                }
                count += 1;
            }
        };
        for (network, names, grads) in [
            (&policy.network, actor_names, &pg),
            (&value.network, value_names, &vg),
        ] {
            for (layer, name) in network.layers.iter().zip(names) {
                compare(
                    to_vec(grads.get::<2>(layer.weight.id).unwrap()).unwrap(),
                    as_f32(&get(&format!("{name}.weight"))["after"]),
                );
                compare(
                    to_vec(grads.get::<1>(layer.bias.as_ref().unwrap().id).unwrap()).unwrap(),
                    as_f32(&get(&format!("{name}.bias"))["after"]),
                );
            }
        }
        compare(
            to_vec(pg.get::<1>(policy.log_std.id).unwrap()).unwrap(),
            as_f32(&get("log_std")["after"]),
        );
        assert_eq!(count, 2371);
        println!(
            "SB3 norm {:.12}, factor {:.12}; Rust norm {norm:.12}, factor {factor:.12}; elements {count}, max abs {max_error:e}, max rel {max_relative:e}",
            reference_norm, reference_factor
        );
    }

    #[test]
    fn joint_norm_scales_all_present_parameters_together() {
        let device = crate::nn::device();
        let policy = LinearConfig::new(1, 1).init(&device);
        let value = LinearConfig::new(1, 1).init(&device);
        let mut pg = GradientsParams::new();
        let mut vg = GradientsParams::new();
        pg.register(
            policy.weight.id,
            Tensor::<2>::from_data(TensorData::from([[3.0]]), &device),
        );
        vg.register(
            value.weight.id,
            Tensor::<2>::from_data(TensorData::from([[4.0]]), &device),
        );
        assert_eq!(clip(&policy, &value, &mut pg, &mut vg, 1.).unwrap(), 5.);
        assert!(
            (pg.get::<2>(policy.weight.id).unwrap().into_scalar::<f32>() - 3. / 5.000001).abs()
                < 1e-6
        );
        assert!(
            (vg.get::<2>(value.weight.id).unwrap().into_scalar::<f32>() - 4. / 5.000001).abs()
                < 1e-6
        );
    }
}
