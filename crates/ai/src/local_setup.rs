//! One way to prefer local servers, for the command line, the window and MCP.

use crate::{AiConfig, BackendConfig, BackendEntry, BackendKind, BackendOptions, detect, openai};
use craft_anything_engine::CancelToken;
use craft_anything_engine::tool::{Ctx, Effects, Requires, ToolOutput, ToolSpec, Toolset, parse_params, schema_of};
use craft_anything_evidence::{ErrorCode, ToolError};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;

/// Register the local-server setup. The command line, the window and MCP all call this.
pub fn specs() -> Vec<ToolSpec> {
    let effects = Effects::LAUNCHES_PROCESS.union(Effects::ACCESSES_NETWORK).union(Effects::WRITES_FILESYSTEM);
    vec![ToolSpec {
        id: "ai_use_local",
        title: "Use local servers",
        summary: "Look for Lemonade and LM Studio at the given addresses, add each that answers with one chat model, and use the first for the lead, worker and builder roles. The rest, then an optional Grok Build program, form the fallback chain. The program is checked with --version before anything is saved. A dry run changes nothing.",
        beginner: "Adds the local servers that are running and uses the first one.",
        toolset: Toolset::Setup,
        effects,
        requires: Requires::Nothing,
        input_schema: schema_of::<Params>,
        run: ai_use_local,
    }]
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Lemonade Server's address.
    #[serde(default = "lemonade_address")]
    lemonade: String,
    /// LM Studio's address.
    #[serde(default = "lmstudio_address")]
    lmstudio: String,
    /// Grok Build's program, added last in the fallback chain.
    grok: Option<String>,
    /// Report what was found and change nothing.
    #[serde(default)]
    dry_run: bool,
}

fn lemonade_address() -> String {
    "http://127.0.0.1:13305/v1".into()
}

fn lmstudio_address() -> String {
    "http://127.0.0.1:1234/v1".into()
}

/// What [`prepare`] was asked to do.
pub struct Request {
    pub lemonade: String,
    pub lmstudio: String,
    pub grok: Option<String>,
    pub dry_run: bool,
}

/// Look, and when this is not a dry run, the settings to save. Nothing is written here.
pub fn prepare(cfg: &AiConfig, req: &Request, cancel: &CancelToken) -> Result<(Option<AiConfig>, Value), ToolError> {
    let grok = req.grok.as_deref().map(str::trim).filter(|program| !program.is_empty());
    if let Some(program) = grok {
        crate::cli_agents::adapters::require_grok_program(program)?;
    }
    let mut found = Vec::new();
    for (name, url) in [("Lemonade", req.lemonade.as_str()), ("LM Studio", req.lmstudio.as_str())] {
        cancel.check()?;
        let url = url.trim().trim_end_matches('/');
        if detect::port_open(url, Duration::from_millis(700)) {
            found.push((name, url.to_string()));
        }
    }
    let found_json = found.iter().map(|(name, url)| json!({"name": name, "base_url": url})).collect::<Vec<_>>();
    if req.dry_run {
        return Ok((None, json!({"found": found_json, "grok": grok, "changed": false})));
    }
    let mut next = cfg.clone();
    let mut servers = Vec::new();
    let mut notes = Vec::new();
    for (name, url) in &found {
        match server_model(name, url, cancel) {
            Ok(model) => {
                next.upsert(BackendEntry {
                    config: BackendConfig {
                        name: (*name).into(),
                        kind: BackendKind::OpenaiCompatible,
                        model,
                        base_url: Some(url.clone()),
                        ssh_host: None,
                        enabled: true,
                    },
                    options: BackendOptions::default(),
                });
                servers.push((*name).to_string());
            }
            Err(e) => notes.push(format!("{name}: {}", e.message)),
        }
    }
    let Some(first) = servers.first().cloned() else {
        return Err(ToolError::new(ErrorCode::BadSettings, "No local server answered.")
            .with_remediation("Start Lemonade (lemond) or LM Studio's server, then run this again."));
    };
    let mut fallback: Vec<String> = servers.iter().skip(1).cloned().collect();
    if let Some(program) = grok {
        next.upsert(BackendEntry {
            config: BackendConfig {
                name: "Grok Build".into(),
                kind: BackendKind::AgentCli,
                model: "grok".into(),
                base_url: None,
                ssh_host: None,
                enabled: true,
            },
            options: BackendOptions { command: Some(program.to_string()), acp: Some(true), ..Default::default() },
        });
        fallback.push("Grok Build".into());
    }
    for role in ["lead", "worker", "builder"] {
        next.roles.insert(role.into(), first.clone());
    }
    next.fallback_chain = fallback;
    let value = json!({"servers": servers, "roles": next.roles, "fallback_chain": next.fallback_chain, "notes": notes, "changed": true});
    Ok((Some(next), value))
}

fn server_model(name: &str, address: &str, cancel: &CancelToken) -> Result<String, ToolError> {
    cancel.check()?;
    let client = openai::OpenAi::compatible(name, address, None, "");
    let models = client.list_models_within(Duration::from_secs(2), Duration::from_secs(5))?;
    cancel.check()?;
    if let Some(loaded) = detect::chat_model(&client.loaded_models(Duration::from_secs(1))) {
        return Ok(loaded);
    }
    match detect::chat_models(&models).as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(ToolError::new(ErrorCode::NotFound, format!("{address} doesn't list a model that can chat."))
            .with_remediation("Load a chat model on the server first.")),
        many => Err(ToolError::new(ErrorCode::BadInput, format!("{address} lists {} chat models and doesn't say which one is loaded.", many.len()))
            .with_remediation("Name the model when you add this server.")),
    }
}

fn ai_use_local(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: Params = parse_params(params)?;
    let cfg = AiConfig::from_table(&ctx.session.settings.ai)?;
    let req = Request { lemonade: p.lemonade, lmstudio: p.lmstudio, grok: p.grok, dry_run: p.dry_run };
    let (next, value) = prepare(&cfg, &req, &ctx.cancel)?;
    if let Some(next) = next {
        let path = ctx.session.config_dir().join("config.toml");
        let mut settings = ctx.session.settings.clone();
        settings.ai = next.to_table()?;
        settings.save_to(&path)?;
        ctx.session.settings = settings;
    }
    Ok(ToolOutput::Value(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use craft_anything_engine::{CallOptions, Caller, Consent, Engine};

    fn call(params: Value) -> (std::sync::Arc<Engine>, Result<Value, ToolError>) {
        let engine = Engine::shared(craft_anything_engine::testing::session(), crate::tools());
        let result =
            engine.call("ai_use_local", params, CallOptions::new(Caller::Cli).with_consent(Consent::Granted)).map(|out| out.to_json()["result"].clone());
        (engine, result)
    }

    #[test]
    fn a_server_that_answers_is_saved_and_used_for_every_role() {
        let url = crate::tests::serve("200 OK", r#"{"object":"list","data":[{"id":"gemma"}]}"#);
        let (engine, result) = call(json!({"lemonade": url, "lmstudio": "http://127.0.0.1:9/v1"}));
        let value = result.unwrap();
        assert_eq!(value["servers"][0], "Lemonade");
        assert_eq!(value["roles"]["lead"], "Lemonade");
        assert_eq!(value["roles"]["worker"], "Lemonade");
        assert_eq!(value["roles"]["builder"], "Lemonade");
        assert_eq!(value["changed"], true);
        let cfg = AiConfig::from_settings(&engine.session().settings);
        assert_eq!(cfg.get("Lemonade").map(|b| b.config.model.as_str()), Some("gemma"));
        assert!(cfg.get("LM Studio").is_none(), "a closed port is not added");
    }

    #[test]
    fn a_dry_run_writes_nothing() {
        let engine = Engine::shared(craft_anything_engine::testing::session(), crate::tools());
        let before = engine.session().settings.ai.clone();
        let result = engine
            .call(
                "ai_use_local",
                json!({"dry_run": true, "lemonade": "http://127.0.0.1:9/v1", "lmstudio": "http://127.0.0.1:9/v1"}),
                CallOptions::new(Caller::Cli).with_consent(Consent::Granted),
            )
            .unwrap();
        assert_eq!(result.to_json()["result"]["changed"], false);
        assert!(result.to_json()["result"]["found"].as_array().is_some_and(|found| found.is_empty()));
        assert_eq!(engine.session().settings.ai, before);
    }

    /// The program is checked first, so a wrong one leaves the server that answered unsaved.
    #[cfg(unix)]
    #[test]
    fn a_program_that_is_not_grok_is_refused_before_anything_is_saved() {
        use std::os::unix::fs::PermissionsExt;
        let url = crate::tests::serve("200 OK", r#"{"object":"list","data":[{"id":"gemma"}]}"#);
        let dir = tempfile::tempdir().unwrap();
        let program = dir.path().join("agent");
        std::fs::write(&program, "#!/bin/sh\nprintf '%s\\n' 'some other program 1.0'\n").unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
        let (engine, result) = call(json!({"lemonade": url, "grok": program}));
        let err = result.expect_err("the program is not Grok Build");
        assert!(err.message.contains("doesn't say"), "{}", err.message);
        assert!(AiConfig::from_settings(&engine.session().settings).get("Lemonade").is_none());
    }
}
