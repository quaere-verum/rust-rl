//! Built-in dense neural models composed into diagnostic learner adapters.
mod candidates;
mod conventional;
mod reppo;
mod sac;
use super::{
    activation_inputs::{ActivationProbeInput, mlp_activations},
    evaluation::{DeterministicActor, StochasticActor},
    models::{
        ModelLimits, SnapshotBuilder, categorical_activations, categorical_prediction,
        categorical_prediction_bytes,
    },
    network_geometry::NumericalMlp,
    probes::{self, ProbeSet},
    reference::{ActionGradient, GradientMethod},
    schema::*,
    session::{CaptureRequest, DiagnosticLearner, LearnerInfo},
    value::{
        self, ContinuationIdentity, CriticDeclaration, MomentUtility, ReturnTarget, TwinMembership,
        ValueCriticKind, ValueDeclaration,
    },
};
use crate::{
    Result,
    algorithms::{
        categorical_dpg::CategoricalDpg,
        continuous::{ContinuousConfig, smooth_target_actions},
        d4pg::D4pg,
        ddqn::{Ddqn, diagnostic_greedy_actor},
        md4pg::Md4pg,
        ppo::Ppo,
        td3::Td3,
    },
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        ensure,
        env::{Observation, ObservationBatch},
        learner::Learner,
    },
    nn::{
        self,
        critics::{
            categorical::CategoricalQ,
            moments::Moments,
            quantile::QuantileGrid,
            residual::ResidualQ,
            scalar::{ActionValue, DiscreteQ, QValue, Value},
        },
        gaussian_policy::GaussianPolicy,
        network::{Mlp, Network},
        policy::{Deterministic, DeterministicPolicy, Policy, StochasticPolicy},
        probes::{
            CategoricalReturnProbe, CriticProbe, DiscreteProbe, MomentsProbe, QuantileProbe,
            ScalarProbe, StateValueProbe,
        },
    },
    objectives::{categorical::CategoricalSupport, hl_gauss::HlGaussConfig},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

enum ActorModel<'a> {
    Stochastic(&'a Policy),
    StateGaussian(&'a GaussianPolicy),
    Deterministic(&'a Deterministic),
    MaskedGreedy(&'a DiscreteQ, &'a dyn Actor),
}

impl ActorModel<'_> {
    fn is_stochastic(&self) -> bool {
        matches!(self, Self::Stochastic(_) | Self::StateGaussian(_))
    }
    fn network(&self) -> &Mlp {
        match self {
            Self::Stochastic(model) => &model.network,
            Self::StateGaussian(model) => &model.network,
            Self::Deterministic(model) => &model.network,
            Self::MaskedGreedy(model, _) => &model.network,
        }
    }

    fn space(&self) -> &ActionSpace {
        match self {
            Self::Stochastic(model) => model.action_space(),
            Self::StateGaussian(model) => model.action_space(),
            Self::Deterministic(model) => model.action_space(),
            Self::MaskedGreedy(_, actor) => actor.action_space(),
        }
    }

    fn snapshot(&self, builder: &mut SnapshotBuilder, name: &str) -> Result<()> {
        match self {
            Self::Stochastic(model) => builder.policy(name, model),
            Self::StateGaussian(model) => builder.state_gaussian(name, model),
            Self::Deterministic(model) => builder.deterministic(name, model),
            Self::MaskedGreedy(model, _) => builder.discrete(name, model),
        }
    }
}

enum CriticModel<'a> {
    State(&'a Value),
    Discrete(&'a DiscreteQ),
    Scalar(&'a QValue),
    Quantiles(&'a QValue, &'a QuantileGrid),
    Moments(&'a Moments<QValue>, MomentUtility),
    Categorical(&'a CategoricalQ, &'a HlGaussConfig, bool),
    CategoricalReturn(&'a QValue, &'a CategoricalSupport),
    ResidualScalar(&'a ResidualQ),
    ResidualQuantiles(&'a ResidualQ, &'a QuantileGrid),
    ResidualMoments(&'a Moments<ResidualQ>, MomentUtility),
}

trait DiagnosticContinuousQ: ActionValue + burn::module::ModuleDisplay {
    fn diagnostic_scalar(&self) -> CriticModel<'_>;
    fn diagnostic_quantiles<'a>(&'a self, grid: &'a QuantileGrid) -> CriticModel<'a>;
    fn diagnostic_moments(models: &Moments<Self>, utility: MomentUtility) -> CriticModel<'_>;
    fn diagnostic_gradient(
        &self,
        observation: &Observation,
        action: &Action,
        space: &ActionSpace,
    ) -> Result<Option<ActionGradient>>;
}

impl DiagnosticContinuousQ for QValue {
    fn diagnostic_scalar(&self) -> CriticModel<'_> {
        CriticModel::Scalar(self)
    }
    fn diagnostic_quantiles<'a>(&'a self, grid: &'a QuantileGrid) -> CriticModel<'a> {
        CriticModel::Quantiles(self, grid)
    }
    fn diagnostic_moments(models: &Moments<Self>, utility: MomentUtility) -> CriticModel<'_> {
        CriticModel::Moments(models, utility)
    }
    fn diagnostic_gradient(
        &self,
        observation: &Observation,
        action: &Action,
        space: &ActionSpace,
    ) -> Result<Option<ActionGradient>> {
        scalar_action_gradient(self, observation, action, space)
    }
}

impl DiagnosticContinuousQ for ResidualQ {
    fn diagnostic_scalar(&self) -> CriticModel<'_> {
        CriticModel::ResidualScalar(self)
    }
    fn diagnostic_quantiles<'a>(&'a self, grid: &'a QuantileGrid) -> CriticModel<'a> {
        CriticModel::ResidualQuantiles(self, grid)
    }
    fn diagnostic_moments(models: &Moments<Self>, utility: MomentUtility) -> CriticModel<'_> {
        CriticModel::ResidualMoments(models, utility)
    }
    fn diagnostic_gradient(
        &self,
        observation: &Observation,
        action: &Action,
        space: &ActionSpace,
    ) -> Result<Option<ActionGradient>> {
        residual_action_gradient(self, observation, action, space)
    }
}

impl CriticModel<'_> {
    fn kind(&self) -> ValueCriticKind {
        match self {
            Self::State(_) => ValueCriticKind::StateValue,
            Self::Discrete(_) => ValueCriticKind::DiscreteQ,
            Self::Scalar(_) | Self::ResidualScalar(_) => ValueCriticKind::ScalarQ,
            Self::Quantiles(_, _) | Self::ResidualQuantiles(_, _) => ValueCriticKind::QuantileQ,
            Self::Moments(_, _) | Self::ResidualMoments(_, _) => ValueCriticKind::MomentQ,
            Self::Categorical(..) => ValueCriticKind::CategoricalQ,
            Self::CategoricalReturn(..) => ValueCriticKind::CategoricalReturnQ,
        }
    }

    fn networks(&self, name: &str) -> Vec<(String, &Mlp)> {
        match self {
            Self::State(model) => vec![(name.into(), &model.network)],
            Self::Discrete(model) => vec![(name.into(), &model.network)],
            Self::Scalar(model) | Self::Quantiles(model, _) | Self::CategoricalReturn(model, _) => {
                vec![(name.into(), &model.network)]
            }
            Self::Moments(models, _) => vec![
                (format!("{name}.first"), &models.first.network),
                (format!("{name}.second"), &models.second.network),
            ],
            Self::Categorical(..)
            | Self::ResidualScalar(_)
            | Self::ResidualQuantiles(..)
            | Self::ResidualMoments(..) => Vec::new(),
        }
    }

    fn residual_networks(&self, name: &str) -> Vec<(String, &ResidualQ)> {
        match self {
            Self::ResidualScalar(model) | Self::ResidualQuantiles(model, _) => {
                vec![(name.into(), model)]
            }
            Self::ResidualMoments(models, _) => vec![
                (format!("{name}.first"), &models.first),
                (format!("{name}.second"), &models.second),
            ],
            _ => Vec::new(),
        }
    }

    fn snapshot(&self, builder: &mut SnapshotBuilder, name: &str) -> Result<()> {
        match self {
            Self::State(model) => builder.value(name, model),
            Self::Discrete(model) => builder.discrete(name, model),
            Self::Scalar(model) => builder.scalar(name, model),
            Self::Quantiles(model, grid) => builder.quantiles(name, model, grid),
            Self::Moments(models, _) => builder.moments(name, &models.first, &models.second),
            Self::ResidualScalar(model) => builder.residual_scalar(name, model),
            Self::ResidualQuantiles(model, grid) => builder.residual_quantiles(name, model, grid),
            Self::ResidualMoments(models, _) => {
                builder.residual_moments(name, &models.first, &models.second)
            }
            Self::Categorical(model, support, auxiliary) => {
                builder.categorical(name, model, support, *auxiliary)
            }
            Self::CategoricalReturn(model, support) => {
                builder.categorical_return(name, model, support)
            }
        }
    }

    fn action_conditioned(&self) -> bool {
        !matches!(self, Self::State(_) | Self::Discrete(_))
    }

    fn prediction(
        &self,
        context: RecordContext,
        set: &ProbeSet,
        actions: Option<&[Action]>,
        continuation: &ContinuationIdentity,
    ) -> Result<Chunk> {
        let observations = set.observations().into();
        let (prediction, condition) = match self {
            Self::State(model) => (
                StateValueProbe(*model).predict(observations)?,
                ActionCondition::StateValue,
            ),
            Self::Discrete(model) => (
                DiscreteProbe(*model).predict(observations)?,
                ActionCondition::AllDiscreteActions,
            ),
            Self::Scalar(model) => (
                ScalarProbe::new(*model)?.predict(observations, action_tensor(set, actions)?)?,
                action_condition(set),
            ),
            Self::Quantiles(model, grid) => (
                QuantileProbe::new(*model, grid)?
                    .predict(observations, action_tensor(set, actions)?)?,
                action_condition(set),
            ),
            Self::CategoricalReturn(model, support) => (
                CategoricalReturnProbe::new(*model, support)?
                    .predict(observations, action_tensor(set, actions)?)?,
                action_condition(set),
            ),
            Self::Moments(models, _) => (
                MomentsProbe::new(*models)?.predict(observations, action_tensor(set, actions)?)?,
                action_condition(set),
            ),
            Self::ResidualScalar(model) => (
                ScalarProbe::new(*model)?.predict(observations, action_tensor(set, actions)?)?,
                action_condition(set),
            ),
            Self::ResidualQuantiles(model, grid) => (
                QuantileProbe::new(*model, grid)?
                    .predict(observations, action_tensor(set, actions)?)?,
                action_condition(set),
            ),
            Self::ResidualMoments(models, _) => (
                MomentsProbe::new(*models)?.predict(observations, action_tensor(set, actions)?)?,
                action_condition(set),
            ),
            Self::Categorical(model, support, _) => {
                return categorical_prediction(
                    context,
                    model,
                    support,
                    set,
                    actions.ok_or_else(|| {
                        crate::Error::State("categorical critic requires probe actions".into())
                    })?,
                    action_condition(set),
                    Some(continuation),
                );
            }
        };
        probes::critic_for_policy(
            context,
            prediction,
            condition,
            actions,
            &continuation.identity,
            if continuation.stochastic {
                ActMode::Sample
            } else {
                ActMode::Deterministic
            },
        )
    }
}

fn action_condition(set: &ProbeSet) -> ActionCondition {
    if set.fixed_actions().is_some() {
        ActionCondition::Fixed
    } else {
        ActionCondition::CurrentPolicy
    }
}

fn action_tensor(set: &ProbeSet, actions: Option<&[Action]>) -> Result<burn::tensor::Tensor<2>> {
    let actions =
        actions.ok_or_else(|| crate::Error::State("critic requires probe actions".into()))?;
    let values = probes::actions_array(actions)?;
    Ok(nn::matrix(
        values.as_f32()?.to_vec(),
        actions.len(),
        set.action_space().dimensions(),
    ))
}

struct NamedCritic<'a> {
    name: &'static str,
    model: CriticModel<'a>,
    twin: Option<TwinMembership>,
}

impl<'a> NamedCritic<'a> {
    fn new(name: &'static str, model: CriticModel<'a>) -> Self {
        Self {
            name,
            model,
            twin: None,
        }
    }

    fn twin(name: &'static str, model: &'a impl DiagnosticContinuousQ, index: usize) -> Self {
        Self {
            name,
            model: model.diagnostic_scalar(),
            twin: Some(TwinMembership {
                group: "twin_q".into(),
                index,
            }),
        }
    }

    fn declaration(
        &self,
        actor: &ContinuationIdentity,
        continuation: &ContinuationIdentity,
    ) -> CriticDeclaration {
        CriticDeclaration {
            kind: self.model.kind(),
            learned_baseline: (self.model.kind() == ValueCriticKind::StateValue).then_some(true),
            actor: Some(actor.clone()),
            continuation: Some(continuation.clone()),
            twin: self.twin.clone(),
            utility: match &self.model {
                CriticModel::Moments(_, utility) | CriticModel::ResidualMoments(_, utility) => {
                    Some(utility.clone())
                }
                _ => None,
            },
        }
    }
}

struct RoleCapture<'a> {
    role: &'static str,
    policy_name: &'static str,
    policy: ActorModel<'a>,
    critics: &'a [NamedCritic<'a>],
    continuation: ContinuationIdentity,
}

struct Capture<'a, 'b> {
    request: &'a CaptureRequest<'b>,
    records: Vec<Chunk>,
    activation_values: usize,
    categorical_bytes: usize,
    return_target: Option<ReturnTarget>,
    rng: ChaCha8Rng,
}

impl<'a, 'b> Capture<'a, 'b> {
    fn new(request: &'a CaptureRequest<'b>) -> Self {
        Self {
            request,
            records: Vec::new(),
            activation_values: request.max_activation_values,
            categorical_bytes: request
                .max_chunk_bytes
                .min(ModelLimits::default().max_model_bytes),
            return_target: None,
            rng: ChaCha8Rng::seed_from_u64(request.policy_seed),
        }
    }

    fn push_value(
        &mut self,
        mut chunk: Chunk,
        critics: impl IntoIterator<Item = (String, CriticDeclaration)>,
    ) -> Result<()> {
        let identity = &chunk.context.identity;
        let declaration = ValueDeclaration {
            version: 1,
            run_id: identity.run_id.clone(),
            revision: identity.learner_update_revision,
            model_role: identity.model_role.clone(),
            conventions: self.request.conventions.cloned().filter(|_| {
                self.return_target
                    .is_none_or(ReturnTarget::is_environmental)
            }),
            critics: critics.into_iter().collect(),
        };
        value::attach_declaration(&mut chunk, &declaration)?;
        if let Some(target) = self.return_target {
            value::attach_return_target(&mut chunk, target)?;
        }
        self.records.push(chunk);
        Ok(())
    }

    fn activations(
        &mut self,
        context: &RecordContext,
        name: &str,
        network: &Mlp,
        input: ActivationProbeInput<'_>,
        rows: usize,
    ) -> Result<()> {
        let layers = self.activation_layers(network, rows);
        if !layers.is_empty() {
            let layers = layers.iter().map(String::as_str).collect::<Vec<_>>();
            self.records.push(mlp_activations(
                context.clone(),
                name,
                network,
                input,
                &layers,
            )?);
        }
        Ok(())
    }

    fn activation_layers(&mut self, network: &Mlp, rows: usize) -> Vec<String> {
        let mut layers = Vec::new();
        for (i, layer) in network.layers.iter().enumerate() {
            let width = layer.weight.val().dims()[1];
            let count = rows.saturating_mul(width);
            for suffix in ["pre", "post"] {
                if count <= self.activation_values {
                    self.activation_values -= count;
                    layers.push(format!("layers.{i}.{suffix}"));
                }
            }
        }
        layers
    }

    fn limited(&mut self, context: &RecordContext, message: String) {
        self.records.push(Chunk {
            context: context.clone(),
            record: Record::Event(EventRecord {
                category: "capture_admission".into(),
                message,
                severity: "warning".into(),
            }),
            arrays: BTreeMap::new(),
        });
    }

    fn role_with_target(&mut self, role: RoleCapture<'_>, target: ReturnTarget) -> Result<()> {
        let previous = self.return_target.replace(target);
        let result = self.role(role);
        self.return_target = previous;
        result
    }

    fn role(&mut self, role: RoleCapture<'_>) -> Result<()> {
        let mut context = self.request.context.clone();
        context.identity.model_role = role.role.into();
        let actor = if matches!(role.policy, ActorModel::StateGaussian(_)) {
            role.continuation.clone()
        } else {
            policy_identity(&context, role.role, role.policy.is_stochastic())
        };
        if self.request.predictions || self.request.representations {
            for set in self.request.probes {
                context.probe_set = Some(set.id().into());
                self.probe(&context, set, &role, &actor)?;
            }
        }
        if self.request.snapshots {
            let mut snapshot = SnapshotBuilder::new(
                &format!("revision-{}", context.identity.learner_update_revision),
                context,
                role.policy.network().input_dim(),
                role.policy.space().clone(),
            )?;
            role.policy.snapshot(&mut snapshot, role.policy_name)?;
            for critic in role.critics {
                if !matches!(critic.model, CriticModel::Discrete(_))
                    || critic.name != role.policy_name
                {
                    critic.model.snapshot(&mut snapshot, critic.name)?;
                }
            }
            self.push_value(
                snapshot.finish()?,
                role.critics.iter().map(|critic| {
                    (
                        critic.name.into(),
                        critic.declaration(&actor, &role.continuation),
                    )
                }),
            )?;
        }
        Ok(())
    }

    fn probe(
        &mut self,
        context: &RecordContext,
        set: &ProbeSet,
        role: &RoleCapture<'_>,
        actor: &ContinuationIdentity,
    ) -> Result<()> {
        ensure(
            set.action_space() == role.policy.space(),
            "diagnostic probe action space differs from model",
        )?;
        let rows = set.observations().len();
        let needs_actions = set.fixed_actions().is_none()
            && role
                .critics
                .iter()
                .any(|critic| critic.model.action_conditioned())
            && (self.request.predictions || self.activation_values > 0);
        let mut policy_actions = None;
        if self.request.predictions || needs_actions {
            let prediction = match role.policy {
                ActorModel::Stochastic(model) => {
                    probes::policy(context.clone(), model, set, ActMode::Sample, &mut self.rng)?
                }
                ActorModel::StateGaussian(model) => {
                    probes::policy(context.clone(), model, set, ActMode::Sample, &mut self.rng)?
                }
                ActorModel::Deterministic(model) => action_policy(
                    context,
                    set,
                    &probes::continuous_actions(model.actions(set.observations().into())?)?,
                )?,
                ActorModel::MaskedGreedy(_, actor) => action_policy(
                    context,
                    set,
                    &actor.act(set.observations(), ActMode::Deterministic, &mut self.rng)?,
                )?,
            };
            if needs_actions {
                let Record::Policy(policy) = &prediction.record else {
                    unreachable!()
                };
                policy_actions = Some(decode_actions(
                    &prediction.arrays[&policy.actions],
                    set.action_space(),
                )?);
            }
            if self.request.predictions {
                self.records.push(prediction);
            }
        }
        if self.request.representations {
            self.activations(
                context,
                role.policy_name,
                role.policy.network(),
                ActivationProbeInput::Observations(set),
                rows,
            )?;
        }
        for critic in role.critics {
            let actions = if critic.model.action_conditioned() {
                set.fixed_actions().or(policy_actions.as_deref())
            } else {
                None
            };
            let mut predict = self.request.predictions;
            let categorical_size = match critic.model {
                CriticModel::Categorical(_, support, _) => Some((
                    support.bins,
                    categorical_prediction_bytes(rows, support.bins),
                )),
                CriticModel::CategoricalReturn(_, support) => {
                    let width = support.atoms().len();
                    Some((width, probes::categorical_return_bytes(rows, width)))
                }
                _ => None,
            };
            if predict && let Some((width, data_bytes)) = categorical_size {
                let action_bytes =
                    rows.saturating_mul(set.action_space().dimensions().saturating_mul(8));
                let bytes = data_bytes
                    .map(|size| size.saturating_add(action_bytes))
                    .unwrap_or(usize::MAX);
                if width > value::MAX_VALUE_HEADS || bytes > self.categorical_bytes {
                    self.limited(
                        context,
                        format!(
                            "{} categorical predictions exceed capture admission",
                            critic.name
                        ),
                    );
                    predict = false;
                } else {
                    self.categorical_bytes -= bytes;
                }
            }
            if predict {
                let chunk =
                    critic
                        .model
                        .prediction(context.clone(), set, actions, &role.continuation)?;
                self.push_value(
                    chunk,
                    [(
                        critic.name.into(),
                        critic.declaration(actor, &role.continuation),
                    )],
                )?;
            }
            if self.request.representations
                && self.activation_values > 0
                && (critic.name != role.policy_name
                    || !matches!(critic.model, CriticModel::Discrete(_)))
            {
                if let CriticModel::Categorical(model, _, auxiliary) = critic.model {
                    let source = if set.fixed_actions().is_some() {
                        ActivationActionSource::FixedProbe
                    } else {
                        ActivationActionSource::Policy {
                            identity: context.identity.clone(),
                            model: role.policy_name.into(),
                            stochastic: role.policy.is_stochastic(),
                        }
                    };
                    if let Some(chunk) = categorical_activations(
                        context.clone(),
                        critic.name,
                        model,
                        set,
                        actions.ok_or_else(|| {
                            crate::Error::State("categorical activation requires actions".into())
                        })?,
                        source,
                        auxiliary,
                        &mut self.activation_values,
                    )? {
                        self.records.push(chunk);
                    }
                } else {
                    for (name, network) in critic.model.networks(critic.name) {
                        let input = if !critic.model.action_conditioned() {
                            ActivationProbeInput::Observations(set)
                        } else if set.fixed_actions().is_some() {
                            ActivationProbeInput::FixedActions(set)
                        } else {
                            ActivationProbeInput::PolicyActions {
                                probes: set,
                                actions: actions.ok_or_else(|| {
                                    crate::Error::State("activation requires policy actions".into())
                                })?,
                                identity: context.identity.clone(),
                                model: role.policy_name.into(),
                                stochastic: role.policy.is_stochastic(),
                            }
                        };
                        self.activations(context, &name, network, input, rows)?;
                    }
                    for (name, model) in critic.model.residual_networks(critic.name) {
                        let network = &model.inner().network;
                        let layers = self.activation_layers(network, rows);
                        if layers.is_empty() {
                            continue;
                        }
                        let actions = action_tensor(set, actions)?.without_autodiff();
                        let observations = set.observations().into();
                        let input = burn::tensor::Tensor::cat(
                            vec![
                                nn::observations_on_device(
                                    observations,
                                    model.inner().observation_dim,
                                    &actions.device(),
                                )?,
                                model.residual_actions(observations, actions)?,
                            ],
                            1,
                        );
                        let layers = layers.iter().map(String::as_str).collect::<Vec<_>>();
                        // Residual coordinates have their own input convention. The
                        // legacy raw-action activation contract cannot describe it.
                        self.records.push(probes::activations(
                            context.clone(),
                            &name,
                            network,
                            input,
                            &layers,
                        )?);
                    }
                }
            }
        }
        Ok(())
    }
}

fn action_policy(context: &RecordContext, set: &ProbeSet, actions: &[Action]) -> Result<Chunk> {
    let mut arrays = BTreeMap::from([("actions".into(), probes::actions_array(actions)?)]);
    let masks = if let ActionSpace::Discrete { count } = set.action_space() {
        let values = set
            .observations()
            .iter()
            .flat_map(|observation| {
                (0..*count).map(move |action| {
                    u8::from(observation.dense_mask().is_none_or(|mask| mask[action]))
                })
            })
            .collect();
        arrays.insert(
            "action_masks".into(),
            Array::u8(vec![actions.len(), *count], values)?,
        );
        Some("action_masks".into())
    } else {
        None
    };
    Ok(Chunk {
        context: context.clone(),
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks,
            stochastic: false,
        }),
        arrays,
    })
}

fn policy_identity(context: &RecordContext, role: &str, stochastic: bool) -> ContinuationIdentity {
    let identity = &context.identity;
    let suffix = if role == "online" {
        ""
    } else {
        "/target-policy"
    };
    ContinuationIdentity {
        identity: format!(
            "{}@{}{suffix}",
            identity.run_id, identity.learner_update_revision
        ),
        stochastic,
    }
}

fn target_continuation(
    context: &RecordContext,
    target_noise: f32,
    noise_clip: f32,
) -> ContinuationIdentity {
    let mut continuation = policy_identity(context, "target", false);
    if target_noise > 0. && noise_clip > 0. {
        continuation.identity.push_str(&format!(
            "/Gaussian-smoothing(std={target_noise},clip={noise_clip})"
        ));
        continuation.stochastic = true;
    }
    continuation
}

struct ConditionalTargetActor {
    policy: DeterministicActor<Deterministic>,
    standard_deviation: f32,
    noise_clip: f32,
}

impl Actor for ConditionalTargetActor {
    fn action_space(&self) -> &ActionSpace {
        self.policy.action_space()
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        self.policy.act_into(observations, mode, rng, actions)?;
        if mode == ActMode::Sample && self.standard_deviation > 0. && self.noise_clip > 0. {
            for action in actions {
                let Action::Continuous(values) = action else {
                    return Err(crate::Error::State(
                        "deterministic target predicted a noncontinuous action".into(),
                    ));
                };
                smooth_target_actions(
                    values,
                    self.action_space(),
                    self.standard_deviation,
                    self.noise_clip,
                    rng,
                );
            }
        }
        Ok(())
    }
}

fn conditional_target_actor(
    policy: &Deterministic,
    training: &ContinuousConfig,
    context: &RecordContext,
) -> (Box<dyn Actor>, ContinuationIdentity) {
    let continuation = target_continuation(context, training.target_noise, training.noise_clip);
    let actor = ConditionalTargetActor {
        policy: DeterministicActor::new(policy),
        standard_deviation: training.target_noise,
        noise_clip: training.noise_clip,
    };
    (Box::new(actor), continuation)
}
fn scalar_action_gradient(
    critic: &QValue,
    observation: &Observation,
    action: &Action,
    space: &ActionSpace,
) -> Result<Option<ActionGradient>> {
    observation.validate(critic.observation_dim, space, true)?;
    observation.check_action(space, action)?;
    let Action::Continuous(action) = action else {
        return Ok(None);
    };
    // Bound the numerical trace by the action dimension and total multiplies.
    trace_scalar_action_gradient(critic, observation, action, None)
}

pub(super) fn residual_action_gradient(
    critic: &crate::nn::critics::residual::ResidualQ,
    observation: &Observation,
    action: &Action,
    space: &ActionSpace,
) -> Result<Option<ActionGradient>> {
    observation.validate(critic.inner().observation_dim, space, true)?;
    observation.check_action(space, action)?;
    let Action::Continuous(action) = action else {
        return Ok(None);
    };
    if action.len() > 64 {
        return Ok(None);
    }
    let inputs = std::slice::from_ref(observation);
    let residual = crate::nn::to_vec(critic.residual_actions(
        inputs.into(),
        crate::nn::matrix_on_device(
            action.clone(),
            1,
            action.len(),
            &burn::tensor::Device::flex(),
        ),
    )?)?;
    let scales = critic.scales()?;
    trace_scalar_action_gradient(critic.inner(), observation, &residual, Some(&scales))
}

fn trace_scalar_action_gradient(
    critic: &QValue,
    observation: &Observation,
    action: &[f32],
    scales: Option<&[f32]>,
) -> Result<Option<ActionGradient>> {
    if action.len() > 64 {
        return Ok(None);
    }
    let weights = critic.network.layers.iter().fold(0usize, |sum, layer| {
        let [input, output] = layer.weight.val().dims();
        sum.saturating_add(input.saturating_mul(output))
    });
    if weights.saturating_mul(action.len() + 1) > 16 * 1024 * 1024 {
        return Ok(None);
    }
    let network = NumericalMlp::from_model(&critic.network)?;
    let input = observation
        .dense_values()
        .iter()
        .chain(action)
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    let directions = (0..action.len())
        .map(|dimension| {
            let mut tangent = vec![0.; input.len()];
            tangent[critic.observation_dim + dimension] =
                scales.map_or(1., |scale| f64::from(scale[dimension]).recip());
            tangent
        })
        .collect::<Vec<_>>();
    let layers = network.trace_directions(&input, &directions)?;
    if layers.iter().any(|layer| layer.relu_kinks > 0) {
        return Ok(None);
    }
    let output = layers.last().expect("numerical network has an output");
    Ok(Some(ActionGradient {
        values: output
            .tangents
            .iter()
            .map(|direction| direction[0])
            .collect(),
        method: GradientMethod::Analytic,
    }))
}
fn learner_info(
    algorithm: &'static str,
    configuration: impl serde::Serialize,
) -> Result<LearnerInfo> {
    Ok(LearnerInfo {
        algorithm,
        model: "rust-rl/builtin-mlp/1",
        configuration: serde_json::to_value(configuration)?,
        discount: None,
    })
}

impl DiagnosticLearner for Ppo {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info(
            "PPO",
            serde_json::json!({
                "training": self.config(),
                "update_settings": self.update_settings(),
            }),
        )
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(StochasticActor::new(&self.policy())))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        capture.role(RoleCapture {
            role: "online",
            policy_name: "policy",
            policy: ActorModel::Stochastic(&self.policy()),
            critics: &[NamedCritic::new("value", CriticModel::State(&self.value()))],
            continuation: policy_identity(request.context, "online", true),
        })?;
        Ok(capture.records)
    }
}

impl DiagnosticLearner for Ddqn {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info("DDQN", self.config())
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(diagnostic_greedy_actor(self.policy())))
    }

    fn diagnostic_evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }
    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        for (role, name, model) in [
            ("online", "online_q", self.policy()),
            ("target", "target_q", self.target()),
        ] {
            let actor = diagnostic_greedy_actor(nn::target::frozen_copy(&model));
            capture.role(RoleCapture {
                role,
                policy_name: name,
                policy: ActorModel::MaskedGreedy(&model, &actor),
                critics: &[NamedCritic::new(name, CriticModel::Discrete(&model))],
                continuation: ContinuationIdentity {
                    identity: "optimality".into(),
                    stochastic: false,
                },
            })?;
        }
        Ok(capture.records)
    }
}

impl<Q: DiagnosticContinuousQ> DiagnosticLearner for Td3<Deterministic, Q> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info("TD3", self.training_config())
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(DeterministicActor::new(&self.policy())))
    }

    fn diagnostic_evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn diagnostic_action_gradient(
        &self,
        name: &str,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<ActionGradient>> {
        let critic = match name {
            "q1" => self.critic(),
            "q2" => self.second_critic(),
            _ => return Ok(None),
        };
        critic.diagnostic_gradient(observation, action, self.actor().action_space())
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        Ok(Some(conditional_target_actor(
            &self.target_policy(),
            self.training_config(),
            context,
        )))
    }
    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        let training = self.training_config();
        let continuation =
            target_continuation(request.context, training.target_noise, training.noise_clip);
        for (role, name, policy, first, second) in [
            (
                "online",
                "policy",
                self.policy(),
                self.critic(),
                self.second_critic(),
            ),
            (
                "target",
                "target_policy",
                self.target_policy(),
                self.target_critic(),
                self.target_second_critic(),
            ),
        ] {
            capture.role(RoleCapture {
                role,
                policy_name: name,
                policy: ActorModel::Deterministic(&policy),
                critics: &[
                    NamedCritic::twin("q1", &first, 0),
                    NamedCritic::twin("q2", &second, 1),
                ],
                continuation: continuation.clone(),
            })?;
        }
        Ok(capture.records)
    }
}

impl<Q: DiagnosticContinuousQ> DiagnosticLearner for D4pg<Deterministic, Q> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info(
            "D4PG",
            serde_json::json!({
                "training": self.training_config(),
                "grid": self.grid(),
                "loss": self.loss(),
                "risk": self.risk(),
                "actor_quantile_order": "ascending heads with declared grid masses",
            }),
        )
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(DeterministicActor::new(&self.policy())))
    }

    fn diagnostic_evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        Ok(Some(conditional_target_actor(
            &self.target_policy(),
            self.training_config(),
            context,
        )))
    }
    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        let training = self.training_config();
        let continuation =
            target_continuation(request.context, training.target_noise, training.noise_clip);
        for (role, name, policy, critic) in [
            ("online", "policy", self.policy(), self.critic()),
            (
                "target",
                "target_policy",
                self.target_policy(),
                self.target_critic(),
            ),
        ] {
            capture.role(RoleCapture {
                role,
                policy_name: name,
                policy: ActorModel::Deterministic(&policy),
                critics: &[NamedCritic::new(
                    "quantile_q",
                    critic.diagnostic_quantiles(self.grid()),
                )],
                continuation: continuation.clone(),
            })?;
        }
        Ok(capture.records)
    }
}

impl DiagnosticLearner for CategoricalDpg {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info(
            "Categorical DPG",
            serde_json::json!({
                "training": self.training_config(),
                "support": self.support(),
                "actor_risk_objective": self.risk(),
            }),
        )
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(DeterministicActor::new(&self.policy())))
    }

    fn diagnostic_evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        Ok(Some(conditional_target_actor(
            &self.target_policy(),
            self.training_config(),
            context,
        )))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        let training = self.training_config();
        let continuation =
            target_continuation(request.context, training.target_noise, training.noise_clip);
        for (role, name, policy, critic) in [
            ("online", "policy", self.policy(), self.critic()),
            (
                "target",
                "target_policy",
                self.target_policy(),
                self.target_critic(),
            ),
        ] {
            capture.role_with_target(
                RoleCapture {
                    role,
                    policy_name: name,
                    policy: ActorModel::Deterministic(&policy),
                    critics: &[NamedCritic::new(
                        "categorical_return_q",
                        CriticModel::CategoricalReturn(&critic, self.support()),
                    )],
                    continuation: continuation.clone(),
                },
                ReturnTarget::Environmental,
            )?;
        }
        Ok(capture.records)
    }
}

impl<Q: DiagnosticContinuousQ> DiagnosticLearner for Md4pg<Deterministic, Q> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info(
            "MD4PG",
            serde_json::json!({
                "training": self.training_config(),
                "risk": self.risk(),
                "critic_outputs": ["expected return", "raw second moment"],
            }),
        )
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(DeterministicActor::new(&self.policy())))
    }

    fn diagnostic_evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        Ok(Some(conditional_target_actor(
            &self.target_policy(),
            self.training_config(),
            context,
        )))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        use crate::objectives::risk::Risk;
        let utility = match self.risk() {
            Risk::Expected => MomentUtility::ExpectedReturn,
            Risk::MeanVariance { penalty } => MomentUtility::MeanVariance { penalty: *penalty },
            Risk::Cvar { .. } => MomentUtility::Unsupported {
                name: "CVaR requires quantiles".into(),
            },
        };
        let mut capture = Capture::new(request);
        let training = self.training_config();
        let continuation =
            target_continuation(request.context, training.target_noise, training.noise_clip);
        for (role, name, policy, moments) in [
            (
                "online",
                "policy",
                self.policy(),
                Moments {
                    first: self.critic(),
                    second: self.second_critic(),
                },
            ),
            (
                "target",
                "target_policy",
                self.target_policy(),
                Moments {
                    first: self.target_critic(),
                    second: self.target_second_critic(),
                },
            ),
        ] {
            capture.role(RoleCapture {
                role,
                policy_name: name,
                policy: ActorModel::Deterministic(&policy),
                critics: &[NamedCritic::new(
                    "moment_q",
                    Q::diagnostic_moments(&moments, utility.clone()),
                )],
                continuation: continuation.clone(),
            })?;
        }
        Ok(capture.records)
    }
}
