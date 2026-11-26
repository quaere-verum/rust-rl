use super::{
    ActorObjective, AuxiliaryTargets, Batch, PreparedTargets, RecurrentReppo, RecurrentReppoPolicy,
    ReppoWindow, StructuredPathwiseActor, categorical_actor_terms, gated_actor_objective,
    recurrent_soft_targets,
};
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::data::temporal_on_policy::TemporalLearnerBinding;
use crate::{
    Error, Result,
    algorithms::{
        recurrent_on_policy::{
            distribution::RecurrentPolicyDistribution,
            minibatch::WindowBatch,
            structured::{
                ConditionalBatch, OPERATION_HEAD_ADAM_PATHS, conditional_key,
                conditional_likelihood, conditional_snapshot, evaluate_complete_laws,
                exact_histories, history_inputs, load_structured_models, model_spaces,
            },
        },
        reppo::ReppoConfig,
    },
    core::{
        action::ConditionalAction,
        checkpoint::LearnerRecord,
        ensure,
        env::Observation,
        stateful::StatefulActor,
        stateful_on_policy::{recording_distribution_matches, recording_number_matches},
        temporal::TemporalExecutionMode,
    },
    data::{minibatch::shuffled_indices_into, temporal_on_policy::TemporalBatchKind},
    nn::{
        checkpoint::RecordBurnExt,
        distributions::CategoricalTerms,
        structured::{
            ConditionalPolicyLaw, PackLimits, RecurrentCandidatePrediction,
            RecurrentLightweightConfig, RecurrentLightweightDistributionalQ,
            RecurrentLightweightPolicy, StructuredHistoryInput,
        },
        target::frozen_copy,
        temporal_models::validate_record_model_count,
        to_vec,
    },
};
use burn::tensor::{Device, Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

type StructuredBatch = Batch<StructuredPathwiseActor>;
type StructuredWindow<'a> = ReppoWindow<'a, StructuredPathwiseActor>;
type Q = RecurrentLightweightDistributionalQ;
type Policy = RecurrentLightweightPolicy;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredReppoQArchitecture {
    config: RecurrentLightweightConfig,
    bins: usize,
    predictor: bool,
}

impl StructuredReppoQArchitecture {
    pub(super) fn new(critic: &Q) -> Self {
        Self {
            config: critic.config().clone(),
            bins: critic.output_dim(),
            predictor: critic.supports_auxiliary(),
        }
    }

    pub(super) fn validate_record(&self, record: &LearnerRecord, name: &str) -> Result<()> {
        validate_record_model_count(
            record,
            name,
            Q::parameter_count(&self.config, self.bins, self.predictor)?,
        )
    }

    pub(super) fn load(&self, record: &LearnerRecord, name: &str, device: &Device) -> Result<Q> {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let template = Q::new(
            self.config.clone(),
            self.bins,
            self.predictor,
            &mut rng,
            &Device::flex(),
        )?;
        if device.is_autodiff() {
            record.load_model(name, &template)
        } else {
            record.load_model_inference(name, &template)
        }
    }
}

impl RecurrentReppo<Policy> {
    /// Learn over complete conditional actions with an independent recurrent histogram Q.
    pub fn new_structured(
        policy: Policy,
        critic: Q,
        config: ReppoConfig,
        seed: u64,
    ) -> Result<Self> {
        Self::new_models(policy, critic, config, seed)
    }
}

fn validate_architectures(
    policy: &RecurrentLightweightConfig,
    critic: &StructuredReppoQArchitecture,
    config: &ReppoConfig,
) -> Result<()> {
    Policy::parameter_count(policy)?;
    Q::parameter_count(&critic.config, critic.bins, critic.predictor)?;
    ensure(
        policy.model.input == critic.config.model.input && critic.bins == config.hl_gauss.bins,
        "structured REPPO policy/Q schema or HL-Gauss bins differ",
    )?;
    ensure(
        config.auxiliary_coefficient == 0. || critic.predictor,
        "structured REPPO critic lacks an auxiliary predictor",
    )
}

fn complete_laws(
    policy: &Policy,
    history: &[StructuredHistoryInput<'_>],
    tokens: &[(usize, usize)],
) -> Result<Vec<ConditionalPolicyLaw>> {
    let evaluated = policy.evaluate_histories(history, PackLimits::default())?;
    evaluate_complete_laws(&evaluated, tokens)
}

fn query_laws(
    critic: &Q,
    history: &[StructuredHistoryInput<'_>],
    tokens: &[(usize, usize)],
    laws: &[ConditionalPolicyLaw],
) -> Result<RecurrentCandidatePrediction> {
    let queries: Vec<_> = tokens
        .iter()
        .zip(laws)
        .flat_map(|(&(lane, time), law)| law.keys().iter().map(move |&key| (lane, time, key)))
        .collect();
    critic
        .evaluate_histories_at(history, tokens, PackLimits::default())?
        .query(&queries)
}

fn prepare_expected_targets(
    learner: &RecurrentReppo<Policy>,
    batch: &StructuredBatch,
    window: &StructuredWindow<'_>,
) -> Result<PreparedTargets> {
    let live: Vec<_> = batch
        .rows
        .iter()
        .enumerate()
        .filter(|(_, row)| !batch.sequences.samples[row.sample].steps[row.time].terminated)
        .collect();
    let mut bootstrap = vec![0.; batch.len()];
    let mut entropy = vec![0.; batch.len()];
    let mut auxiliary = None;
    if !live.is_empty() {
        let next: Vec<_> = live
            .iter()
            .map(|(_, row)| (row.sample, row.time + 1))
            .collect();
        let exact = exact_histories(window)?;
        let histories = history_inputs(&exact);
        let laws = complete_laws(batch.behavior.policy(), &histories, &next)?;
        let prediction = query_laws(batch.behavior.critic(), &histories, &next, &laws)?;
        let q = to_vec(learner.config.hl_gauss.decode(prediction.logits)?)?;
        ensure(
            q.iter().all(|x| x.is_finite()),
            "nonfinite structured REPPO successor Q",
        )?;
        let width = prediction.representation.dims()[1];
        let auxiliary_rows = learner.config.auxiliary_coefficient > 0.
            && live
                .iter()
                .any(|(_, row)| !batch.sequences.samples[row.sample].steps[row.time].truncated);
        let representation = auxiliary_rows
            .then(|| to_vec(prediction.representation))
            .transpose()?;
        if let Some(representation) = &representation {
            ensure(
                representation.iter().all(|x| x.is_finite()),
                "nonfinite structured REPPO successor representation",
            )?;
            let size = batch
                .len()
                .checked_mul(width)
                .ok_or_else(|| Error::Shape("structured REPPO auxiliary size overflow".into()))?;
            auxiliary = Some(AuxiliaryTargets {
                values: vec![0.; size],
                width,
            });
        }
        let alpha = f64::from(learner.alpha.value());
        let mut offset = 0;
        for ((i, row), law) in live.into_iter().zip(laws) {
            let logs = to_vec(law.log_probs().clone())?;
            ensure(
                logs.iter().all(|x| x.is_finite()),
                "nonfinite structured REPPO successor log probability",
            )?;
            let probabilities: Vec<_> = logs.iter().map(|&log| f64::from(log).exp()).collect();
            bootstrap[i] = probabilities
                .iter()
                .zip(&q[offset..offset + logs.len()])
                .map(|(&p, &value)| p * f64::from(value))
                .sum::<f64>() as f32;
            entropy[i] = (-alpha
                * probabilities
                    .iter()
                    .zip(&logs)
                    .map(|(&p, &log)| p * f64::from(log))
                    .sum::<f64>()) as f32;
            if let (Some(representation), Some(targets)) = (&representation, &mut auxiliary)
                && !batch.sequences.samples[row.sample].steps[row.time].truncated
            {
                for column in 0..width {
                    targets.values[i * width + column] = probabilities
                        .iter()
                        .enumerate()
                        .map(|(candidate, &p)| {
                            p * f64::from(representation[(offset + candidate) * width + column])
                        })
                        .sum::<f64>()
                        as f32;
                }
            }
            offset += logs.len();
        }
    }
    ensure(
        auxiliary
            .as_ref()
            .is_none_or(|targets| targets.values.iter().all(|x| x.is_finite())),
        "nonfinite structured REPPO expected representation",
    )?;
    let returns = recurrent_soft_targets(batch, &bootstrap, &entropy, learner.config.lambda)?;
    Ok(PreparedTargets { returns, auxiliary })
}

impl RecurrentReppoPolicy for Policy {
    type Critic = Q;
    type Actor = StructuredPathwiseActor;
    type PolicyArchitecture = RecurrentLightweightConfig;
    type CriticArchitecture = StructuredReppoQArchitecture;
    type CriticInput = RecurrentCandidatePrediction;

    const RECORD_VERSION: u32 = 2;
    const SPARSE_ADAM: bool = true;
    const REQUIRED_POLICY_ADAM_PATHS: &'static [&'static str] = OPERATION_HEAD_ADAM_PATHS;
    const REQUIRED_CRITIC_ADAM_PATHS: &'static [&'static str] =
        &["head.layers.1.weight", "head.layers.1.bias"];

    fn architectures(
        &self,
        critic: &Q,
    ) -> Result<(Self::PolicyArchitecture, Self::CriticArchitecture)> {
        Ok((
            self.config().clone(),
            StructuredReppoQArchitecture::new(critic),
        ))
    }

    fn validate_pair(&self, critic: &Q, config: &ReppoConfig) -> Result<()> {
        validate_architectures(
            self.config(),
            &StructuredReppoQArchitecture::new(critic),
            config,
        )
    }

    fn validate_binding(
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
        binding: &TemporalLearnerBinding,
        config: &ReppoConfig,
    ) -> Result<()> {
        validate_architectures(policy, critic, config)?;
        ensure(
            *binding
                == TemporalLearnerBinding::new_structured(
                    TemporalBatchKind::Pathwise,
                    binding.revision,
                    binding.identity,
                    model_spaces(policy),
                )?,
            "structured REPPO collection binding differs from model schema",
        )
    }

    fn validate_record_models(
        record: &LearnerRecord,
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
    ) -> Result<()> {
        validate_record_model_count(record, "policy", Policy::parameter_count(policy)?)?;
        critic.validate_record(record, "critic")
    }

    fn restore_models(
        record: &LearnerRecord,
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
    ) -> Result<(Self, Q)> {
        let device = crate::nn::device();
        let (policy, _) = load_structured_models(record, policy, None, &device)?;
        Ok((policy, critic.load(record, "critic", &device)?))
    }

    fn publish(policy: Self, critic: Q, revision: u64) -> Result<StructuredPathwiseActor> {
        StructuredPathwiseActor::new(policy, critic, revision)
    }

    fn published_models(actor: &StructuredPathwiseActor) -> (&Self, &Q) {
        (actor.policy(), actor.critic())
    }

    fn with_revision(actor: &StructuredPathwiseActor, revision: u64) -> StructuredPathwiseActor {
        actor.with_revision(revision)
    }

    fn binding(actor: &StructuredPathwiseActor) -> Result<TemporalLearnerBinding> {
        TemporalLearnerBinding::new_structured(
            TemporalBatchKind::Pathwise,
            actor.revision(),
            actor.behavior_identity(),
            model_spaces(actor.policy().config()),
        )
    }

    fn windows<'a>(
        &self,
        _: &Q,
        batch: &'a StructuredBatch,
        _: TemporalExecutionMode,
    ) -> Result<WindowBatch<'a, StructuredPathwiseActor>> {
        Ok(WindowBatch::reference(batch))
    }

    fn audit_recordings(batch: &StructuredBatch, window: &StructuredWindow<'_>) -> Result<()> {
        let exact = exact_histories(window)?;
        let tokens = batch.tokens();
        let laws = complete_laws(batch.behavior.policy(), &history_inputs(&exact), &tokens)?;
        for (row, law) in batch.rows.iter().zip(&laws) {
            let sample = &batch.sequences.samples[row.sample];
            let Observation::Structured(frame) = &sample.observations[row.time] else {
                return Err(Error::Shape(
                    "structured REPPO row requires a public frame".into(),
                ));
            };
            let key = conditional_key(&sample.steps[row.time].action)?;
            let log = to_vec(conditional_likelihood(law, key)?)?[0];
            ensure(
                recording_distribution_matches(
                    &conditional_snapshot(law, frame)?,
                    &row.distribution,
                ) && recording_number_matches(log, row.log_prob),
                "structured REPPO collection law differs from frozen history",
            )?;
        }
        Ok(())
    }

    fn prepare_targets(
        learner: &RecurrentReppo<Self>,
        batch: &StructuredBatch,
        window: &StructuredWindow<'_>,
        _: &mut ChaCha8Rng,
    ) -> Result<PreparedTargets> {
        prepare_expected_targets(learner, batch, window)
    }

    fn validate_update_queries(
        learner: &RecurrentReppo<Self>,
        batch: &StructuredBatch,
        windows_per_batch: usize,
        rng: &ChaCha8Rng,
    ) -> Result<()> {
        let mut queries_by_window = vec![0; batch.sequences.samples.len()];
        let mut total = 0usize;
        for row in &batch.rows {
            let observation = &batch.sequences.samples[row.sample].observations[row.time];
            let count = learner.policy.entropy_support(observation).ok_or_else(|| {
                Error::Shape("structured REPPO row requires a public frame".into())
            })?;
            total = total
                .checked_add(count)
                .ok_or_else(|| Error::Shape("structured REPPO query count overflow".into()))?;
            queries_by_window[row.sample] += count;
        }
        let full_budget = learner.critic.validate_query_budget(total);
        if full_budget.is_ok() || windows_per_batch >= queries_by_window.len() {
            return full_budget;
        }

        // Structured actor updates draw no noise, so a copy of the learner's
        // RNG reproduces each scheduled minibatch without advancing the update.
        let mut rng = rng.clone();
        let mut shuffled = Vec::new();
        for _ in 0..learner.config.epochs {
            shuffled_indices_into(
                queries_by_window.len(),
                windows_per_batch,
                &mut rng,
                &mut shuffled,
            )?;
            for windows in shuffled.chunks(windows_per_batch) {
                let count = windows.iter().map(|&index| queries_by_window[index]).sum();
                learner.critic.validate_query_budget(count)?;
            }
        }
        Ok(())
    }

    fn critic_input(
        critic: &Q,
        window: &StructuredWindow<'_>,
        tokens: &[(usize, usize)],
    ) -> Result<RecurrentCandidatePrediction> {
        let exact = exact_histories(window)?;
        let queries = tokens
            .iter()
            .map(|&(lane, time)| {
                Ok((
                    lane,
                    time,
                    conditional_key(&window.sample(lane).steps[time].action)?,
                ))
            })
            .collect::<Result<Vec<(usize, usize, ConditionalAction)>>>()?;
        critic
            .evaluate_histories_at(&history_inputs(&exact), tokens, PackLimits::default())?
            .query(&queries)
    }

    fn critic_logits(_: &Q, input: &RecurrentCandidatePrediction) -> Result<Tensor<2>> {
        Ok(input.logits.clone())
    }

    fn predict_representation(
        critic: &Q,
        input: RecurrentCandidatePrediction,
    ) -> Result<Tensor<2>> {
        critic.predict_representation(input.representation)
    }

    fn actor_for_window(
        learner: &RecurrentReppo<Self>,
        window: &StructuredWindow<'_>,
        tokens: &[(usize, usize)],
        _: &[&Observation],
        _: &mut ChaCha8Rng,
    ) -> Result<ActorObjective> {
        let exact = exact_histories(window)?;
        let history = history_inputs(&exact);
        let current = complete_laws(&learner.policy, &history, tokens)?;
        let behavior = ConditionalBatch::from_snapshots(
            &window.behavior_distributions(),
            &learner.policy.model_device(),
        )?;
        let kl = Tensor::cat(
            behavior
                .complete()?
                .iter()
                .zip(&current)
                .map(|(old, current)| old.kl(current))
                .collect::<Result<_>>()?,
            0,
        );
        let (alpha, beta) = learner.multipliers();
        gated_actor_objective(kl, beta, learner.config.kl_limit, || {
            // The critic has already stepped. Its own history is freshly
            // reconstructed, and complete candidate values are constants here.
            let critic = frozen_copy(&learner.critic);
            let q = learner
                .config
                .hl_gauss
                .decode(query_laws(&critic, &history, tokens, &current)?.logits)?;
            ensure(
                to_vec(q.clone())?.iter().all(|x| x.is_finite()),
                "nonfinite structured REPPO actor Q",
            )?;
            let mut offset = 0;
            let mut ordinary = Vec::with_capacity(current.len());
            let mut entropy = Vec::with_capacity(current.len());
            for law in current {
                let count = law.keys().len();
                let log_probs = law.log_probs().clone().reshape([1, count]);
                let terms = CategoricalTerms {
                    probabilities: log_probs.clone().exp(),
                    log_probs,
                };
                let values = q.clone().narrow(0, offset, count).reshape([1, count]);
                let (loss, row_entropy) = categorical_actor_terms(terms, values, alpha)?;
                ordinary.push(loss);
                entropy.push(row_entropy);
                offset += count;
            }
            Ok((Tensor::cat(ordinary, 0), to_vec(Tensor::cat(entropy, 0))?))
        })
    }

    fn entropy_support(&self, observation: &Observation) -> Option<usize> {
        match observation {
            Observation::Structured(frame) => Some(
                frame
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.enabled)
                    .count(),
            ),
            Observation::Dense(_) | Observation::Discrete(_) | Observation::Graph(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        collect::vector_env::SyncVectorEnv,
        core::{
            action::{Action, ActionArgument, ActionSpace},
            env::{Env, EpisodeStart, ObservationSpace, Step},
            structured::ActionCandidate,
            temporal::TemporalSpaces,
        },
        data::temporal::SequenceConfig,
        objectives::hl_gauss::HlGaussConfig,
        train::{
            config::OnPolicyConfig,
            temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
        },
    };
    use burn::{module::Param, nn::Linear};
    use std::sync::Arc;

    #[allow(dead_code)]
    mod delayed {
        use crate as rl;
        include!("../../../tests/support/delayed_candidate.rs");
    }

    struct TerminalChoice {
        spaces: TemporalSpaces,
        current: Observation,
        terminal: Observation,
    }

    impl TerminalChoice {
        fn new(config: &RecurrentLightweightConfig) -> Self {
            let mut current = delayed::DelayedCandidate::new()
                .reset(Some(1))
                .unwrap()
                .observation;
            let Observation::Structured(frame) = &mut current else {
                unreachable!()
            };
            let frame = Arc::make_mut(frame);
            frame.candidates[0].features[0] = 0.;
            for (index, entity) in frame.entities.iter().enumerate() {
                frame.candidates.push(ActionCandidate {
                    action: ConditionalAction {
                        operation: 1,
                        argument: Some(ActionArgument::Entity(entity.id)),
                    },
                    features: vec![(index + 1) as f32],
                    enabled: index < 2,
                });
            }
            // Current laws follow public order; frozen snapshots follow semantic order.
            frame.candidates.swap(0, 2);
            let mut terminal = frame.clone();
            terminal.candidates.clear();
            Self {
                spaces: model_spaces(config),
                current,
                terminal: Observation::Structured(Arc::new(terminal)),
            }
        }
    }

    impl Env for TerminalChoice {
        fn observation_space(&self) -> ObservationSpace {
            self.spaces.observations.clone()
        }
        fn action_space(&self) -> &ActionSpace {
            &self.spaces.actions
        }
        fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
            Ok(EpisodeStart::fresh(self.current.clone()))
        }
        fn step(&mut self, action: &Action) -> Result<Step> {
            self.current.check_action(&self.spaces.actions, action)?;
            Ok(Step {
                observation: self.terminal.clone(),
                reward: 1f32.into(),
                terminated: true,
                truncated: false,
            })
        }
    }

    fn route_weight(layer: &mut Linear, input: usize, output: usize, value: f32) {
        let previous = layer.weight.val();
        let [inputs, outputs] = previous.dims();
        let mut weights = vec![0.; inputs * outputs];
        weights[input * outputs + output] = value;
        layer.weight = Param::from_tensor(
            Tensor::<1>::from_data(weights.as_slice(), &previous.device())
                .reshape([inputs, outputs]),
        );
    }

    fn set_policy_logits(policy: &mut Policy, operation: f32, argument: f32) {
        let target_width = policy.config().model.target_width();
        let operation_head = &mut policy.operation_head.layers;
        // The last input is ln(1 + enabled candidates in the branch).
        let count_feature = operation_head[0].weight.val().dims()[0] - 1;
        route_weight(&mut operation_head[0], count_feature, 0, 1.);
        route_weight(&mut operation_head[1], 0, 0, operation);
        route_weight(&mut policy.entity_key.layers[0], target_width, 0, 1.);
        let query = &mut policy.entity_query.layers[0];
        let weight = query.weight.val();
        let mut bias = vec![0.; weight.dims()[1]];
        bias[0] = argument;
        query.weight = Param::from_tensor(Tensor::zeros(weight.dims(), &weight.device()));
        query.bias = Some(Param::from_tensor(Tensor::from_data(
            bias.as_slice(),
            &weight.device(),
        )));
    }

    #[test]
    fn complete_action_actor_gradients_match_literal_objective_and_forward_kl_gate() {
        let config = delayed::model_config();
        let device = Device::flex().autodiff();
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let mut policy = Policy::new(config.clone(), &mut rng, &device).unwrap();
        let (old_x, old_y) = (-1.1_f32, -0.9_f32);
        set_policy_logits(&mut policy, old_x, old_y);
        let mut critic = Q::new(config.clone(), 2, false, &mut rng, &device).unwrap();
        let feature =
            config.hidden_width + config.model.operation_width + config.model.target_width();
        // The two histogram logits are [0, candidate feature], on centers [-1, 1].
        route_weight(&mut critic.head.layers[0], feature, 0, 1.);
        route_weight(&mut critic.head.layers[1], 0, 1, 1.);
        let mut learner = RecurrentReppo::new_structured(
            policy,
            critic,
            ReppoConfig {
                initial_alpha: 0.2,
                initial_beta: 1.7,
                hl_gauss: HlGaussConfig {
                    min: -1.,
                    max: 1.,
                    bins: 2,
                    sigma_ratio: 0.75,
                },
                ..Default::default()
            },
            17,
        )
        .unwrap();
        let mut run = TemporalOnPolicyTrainer::new(
            SyncVectorEnv::new(vec![TerminalChoice::new(&config)]).unwrap(),
            &learner,
            TemporalOnPolicyConfig {
                collection: OnPolicyConfig {
                    rollout_steps: 1,
                    gae_lambda: None,
                    ..Default::default()
                },
                sequence: SequenceConfig {
                    learn_len: 1,
                    burn_in: 0,
                    n_step: 1,
                },
                ..Default::default()
            },
            19,
        )
        .unwrap();
        run.collect_tick(&learner).unwrap();
        let window = WindowBatch::reference(run.seal(&learner).unwrap())
            .full()
            .unwrap();
        let tokens = window.tokens();
        let observations = window.observations(&tokens);
        let (x, y) = (0.7_f32, 0.8_f32);
        set_policy_logits(&mut learner.policy, x, y);

        let sigmoid = |logit: f64| 1. / (1. + (-logit).exp());
        let probabilities = |operation, argument| {
            let choose = sigmoid(operation * (3f64 / 2.).ln());
            let right = sigmoid(argument / (config.model.entity_width as f64).sqrt());
            [1. - choose, choose * (1. - right), choose * right]
        };
        let old = probabilities(f64::from(old_x), f64::from(old_y));
        let forward_kl = |current: [f64; 3]| {
            old.into_iter()
                .zip(current)
                .map(|(p, q)| p * (p / q).ln())
                .sum::<f64>()
        };
        let q_values = [0., 1., 2.].map(|feature| 2. * sigmoid(feature) - 1.);
        let (alpha, beta) = learner.multipliers();
        let (x, y) = (f64::from(x), f64::from(y));
        let kl = forward_kl(probabilities(x, y));
        assert!((0.01..0.5).contains(&kl));
        let before = rng.clone();
        for constrained in [false, true] {
            learner.config.kl_limit = if constrained { 0.01 } else { 0.5 };
            let reference = |x, y| {
                let probabilities = probabilities(x, y);
                if constrained {
                    f64::from(beta) * forward_kl(probabilities)
                } else {
                    probabilities
                        .into_iter()
                        .zip(q_values)
                        .map(|(p, q)| p * (f64::from(alpha) * p.ln() - q))
                        .sum::<f64>()
                }
            };
            let result =
                Policy::actor_for_window(&learner, &window, &tokens, &observations, &mut rng)
                    .unwrap();
            assert!((f64::from(result.kl[0]) - kl).abs() < 2e-6);
            let actual = f64::from(result.loss.clone().into_scalar::<f32>());
            assert!((actual - reference(x, y)).abs() < 2e-6);
            let gradients = result.loss.backward();
            let operation = learner.policy.operation_head.layers[1].weight.val();
            let argument = learner.policy.entity_query.layers[0]
                .bias
                .as_ref()
                .unwrap()
                .val();
            let epsilon = 1e-4;
            for (parameter, expected) in [
                (
                    to_vec(operation.grad(&gradients).unwrap()).unwrap()[0],
                    (reference(x + epsilon, y) - reference(x - epsilon, y)) / (2. * epsilon),
                ),
                (
                    to_vec(argument.grad(&gradients).unwrap()).unwrap()[0],
                    (reference(x, y + epsilon) - reference(x, y - epsilon)) / (2. * epsilon),
                ),
            ] {
                assert!(expected.abs() > 1e-3);
                assert!((f64::from(parameter) - expected).abs() < 2e-6);
            }
            assert!(
                learner.critic.head.layers[0]
                    .weight
                    .val()
                    .grad(&gradients)
                    .is_none()
            );
        }
        assert_eq!(rng, before);
    }
}
