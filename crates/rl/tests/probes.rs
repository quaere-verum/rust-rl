#![cfg(feature = "neural")]

use burn::{
    module::{Module, Param},
    tensor::Tensor,
};
use rl::nn::{matrix, network::Network, to_vec};

#[derive(Module, Debug)]
struct CustomNetwork {
    weight: Param<Tensor<2>>,
}
impl Network for CustomNetwork {
    fn input_dim(&self) -> usize {
        1
    }
    fn output_dim(&self) -> usize {
        1
    }
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        input.matmul(self.weight.val())
    }
}

#[test]
fn custom_network_needs_no_probe_implementation() {
    let network = CustomNetwork {
        weight: Param::from_tensor(matrix(vec![2.], 1, 1)),
    };
    assert_eq!(
        to_vec(network.forward(matrix(vec![3.], 1, 1))).unwrap(),
        [6.]
    );
}

#[cfg(feature = "probes")]
mod enabled {
    use super::*;
    use burn::module::RunningState;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rl::{
        Result,
        core::env::{Observation, ObservationBatch},
        nn::{
            critics::{
                moments::Moments,
                quantile::QuantileGrid,
                scalar::{ActionValue, QValue},
            },
            network::Mlp,
            probes::{
                ActivationProbe, CriticPrediction, CriticProbe, MomentsProbe, QuantileProbe,
                ScalarProbe,
            },
            target::frozen_copy,
            vector,
        },
    };

    #[test]
    fn selected_mlp_pre_and_post_activations_have_exact_values_and_no_graph() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut network = Mlp::new(2, &[2], 1, &mut rng).unwrap();
        network.layers[0].weight = Param::from_tensor(matrix(vec![1., -2., 3., 4.], 2, 2));
        network.layers[0].bias = Some(Param::from_tensor(vector(&[0.5, -1.])));
        network.layers[1].weight = Param::from_tensor(matrix(vec![2., -1.], 2, 1));
        network.layers[1].bias = Some(Param::from_tensor(vector(&[1.])));
        let names = [
            "layers.1.post",
            "layers.0.pre",
            "layers.0.post",
            "layers.1.pre",
        ];
        let outputs = network
            .activations(matrix(vec![1., 2., -1., 0.], 2, 2).require_grad(), &names)
            .unwrap();
        let expected = [
            vec![11., 0.],
            vec![7.5, 5., -0.5, 1.],
            vec![7.5, 5., 0., 1.],
            vec![11., 0.],
        ];
        for ((output, name), expected) in outputs.iter().zip(names).zip(expected) {
            assert_eq!(output.name, name);
            assert!(!output.tensor.is_autodiff());
            assert_eq!(to_vec(output.tensor.clone()).unwrap(), expected);
            assert_eq!(output.tensor.dims()[0], 2);
        }
        assert!(
            network
                .activations(matrix(vec![0., 0.], 1, 2), &[])
                .unwrap()
                .is_empty()
        );
        for name in [
            "layers.2.pre",
            "layers.0.unknown",
            "layers.0.pre.extra",
            "layers.00.pre",
        ] {
            assert!(
                network
                    .activations(matrix(vec![0., 0.], 1, 2), &[name])
                    .is_err()
            );
        }
        assert!(
            network
                .activations(matrix(vec![0.], 1, 1), &["layers.0.pre"])
                .is_err()
        );
    }

    fn constant_critic(outputs: &[f32]) -> QValue {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut critic = QValue::new(1, 1, outputs.len(), &[], &mut rng).unwrap();
        critic.network.layers[0].weight =
            Param::from_tensor(matrix(vec![0.; 2 * outputs.len()], 2, outputs.len()));
        critic.network.layers[0].bias = Some(Param::from_tensor(vector(outputs)));
        critic
    }

    #[test]
    fn critic_adapters_distinguish_scalar_quantile_and_raw_moment_shapes() {
        let observations = [Observation::new(vec![0.]), Observation::new(vec![1.])];
        let actions = matrix(vec![0.2, -0.3], 2, 1).require_grad();
        let scalar = constant_critic(&[2.]);
        let CriticPrediction::Scalar(values) = ScalarProbe::new(&scalar)
            .unwrap()
            .predict((&observations).into(), actions.clone())
            .unwrap()
        else {
            panic!("expected scalar prediction")
        };
        assert_eq!(values.dims(), [2]);
        assert!(!values.is_autodiff());
        assert_eq!(to_vec(values).unwrap(), [2., 2.]);

        let quantiles = constant_critic(&[-2., 1., 4.]);
        let grid = QuantileGrid {
            edges: vec![0., 0.2, 0.6, 1.],
        };
        let CriticPrediction::Quantiles {
            values,
            locations,
            masses,
        } = QuantileProbe::new(&quantiles, &grid)
            .unwrap()
            .predict((&observations).into(), actions.clone())
            .unwrap()
        else {
            panic!("expected quantiles")
        };
        assert_eq!(values.dims(), [2, 3]);
        assert!(!values.is_autodiff());
        assert_eq!(to_vec(values).unwrap(), [-2., 1., 4., -2., 1., 4.]);
        for (actual, expected) in locations.iter().zip([0.1, 0.4, 0.8]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in masses.iter().zip([0.2, 0.4, 0.4]) {
            assert!((actual - expected).abs() < 1e-6);
        }

        let moments = Moments {
            first: constant_critic(&[2.]),
            second: constant_critic(&[7.]),
        };
        let CriticPrediction::Moments { mean, raw_second } = MomentsProbe::new(&moments)
            .unwrap()
            .predict((&observations).into(), actions)
            .unwrap()
        else {
            panic!("expected moments")
        };
        assert_eq!(mean.dims(), [2]);
        assert_eq!(raw_second.dims(), [2]);
        assert!(!mean.is_autodiff() && !raw_second.is_autodiff());
        assert_eq!(to_vec(mean).unwrap(), [2., 2.]);
        assert_eq!(to_vec(raw_second).unwrap(), [7., 7.]);
        assert!(ScalarProbe::new(&quantiles).is_err());
        assert!(QuantileProbe::new(&scalar, &grid).is_err());
        assert!(
            QuantileProbe::new(
                &scalar,
                &QuantileGrid {
                    edges: vec![0., 0.]
                }
            )
            .is_err()
        );
    }

    #[test]
    fn probing_early_layers_does_not_forward_later_layers() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut network = Mlp::new(2, &[2], 1, &mut rng).unwrap();
        // A three-row final weight cannot consume the two-wide hidden layer.
        // Reaching that layer would fail, even if its output were discarded.
        network.layers[1].weight = Param::from_tensor(matrix(vec![1.; 3], 3, 1));
        let result = network
            .activations(matrix(vec![1., 2.], 1, 2), &["layers.0.post"])
            .unwrap();
        assert_eq!(result[0].tensor.dims(), [1, 2]);
        network.layers[0].weight = Param::from_tensor(matrix(vec![1.; 6], 3, 2));
        assert!(
            network
                .activations(matrix(vec![1., 2.], 1, 2), &[])
                .unwrap()
                .is_empty()
        );
    }

    #[derive(Module, Debug)]
    struct StatefulCritic {
        running: RunningState<Tensor<1>>,
        weight: Param<Tensor<2>>,
    }
    impl ActionValue for StatefulCritic {
        fn action_dim(&self) -> usize {
            1
        }
        fn output_dim(&self) -> usize {
            1
        }
        fn values(&self, _: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
            let old = self.running.value_sync();
            self.running.update(old.clone() + 1.);
            Ok(actions.matmul(self.weight.val()) + old.reshape([1, 1]))
        }
    }

    #[test]
    fn probes_do_not_mutate_online_or_target_buffers_parameters_or_owned_rng() {
        let mut rng = ChaCha8Rng::seed_from_u64(29);
        let _network = Mlp::new(1, &[2], 1, &mut rng).unwrap();
        let mut expected_rng = rng.clone();
        let critic = StatefulCritic {
            running: RunningState::new(vector(&[3.])),
            weight: Param::from_tensor(matrix(vec![2.], 1, 1)),
        };
        let target = frozen_copy(&critic);
        let observations = [Observation::new(vec![0.])];
        for _ in 0..2 {
            let CriticPrediction::Scalar(values) = ScalarProbe::new(&critic)
                .unwrap()
                .predict(
                    (&observations).into(),
                    matrix(vec![1.], 1, 1).require_grad(),
                )
                .unwrap()
            else {
                panic!("expected scalar prediction")
            };
            assert_eq!(to_vec(values).unwrap(), [5.]);
        }
        assert_eq!(to_vec(critic.running.value_sync()).unwrap(), [3.]);
        assert_eq!(to_vec(target.running.value_sync()).unwrap(), [3.]);
        assert_eq!(to_vec(critic.weight.val()).unwrap(), [2.]);
        assert_eq!(to_vec(target.weight.val()).unwrap(), [2.]);
        assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
    }

    #[test]
    fn critic_probe_rejects_input_and_output_shape_mismatches() {
        let critic = StatefulCritic {
            running: RunningState::new(vector(&[3.])),
            // The public critic contract says one output, but this model emits two.
            weight: Param::from_tensor(matrix(vec![1., 2.], 1, 2)),
        };
        let observations = [Observation::new(vec![0.])];
        let probe = ScalarProbe::new(&critic).unwrap();
        assert!(
            probe
                .predict((&observations).into(), matrix(vec![1.], 1, 1))
                .is_err()
        );
        assert!(
            probe
                .predict((&observations).into(), matrix(vec![1., 2.], 1, 2))
                .is_err()
        );
        assert_eq!(to_vec(critic.running.value_sync()).unwrap(), [3.]);
    }
}
