# Original schema-3 traffic trajectory

`traffic_schema3_trajectory.rld` is an unchanged copy of the prior writer's chunk:

- Run: `traffic-ddqn-seed7`, training seed **7**, schema **3**.
- Recorded episode: `never_switch-k0-seed30000`, evaluation scenario seed **30000**, revision **0**, role `reference-never_switch`.
- Payload: 64 transitions, 65 observations and action masks, complete terminal episode; no render metadata.
- Original size: **4,446 bytes** (compressed RLDIAG01 chunk).
- SHA-256: `93b91a6a57cdbe9de920aba4d40aba51f292deeb27b571db22d21dd316cde34c`.

The codec compatibility test decodes the original writer's unmodified bytes. Retain this fixture when changing diagnostic schema support.
