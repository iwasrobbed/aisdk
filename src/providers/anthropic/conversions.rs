use crate::core::Message;
use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponseContentType, ReasoningEffort, Usage,
};
use crate::providers::anthropic::client::{
    AnthropicAssistantMessageParamContent, AnthropicCacheControl, AnthropicMessageDeltaUsage,
    AnthropicMessageParam, AnthropicOptions, AnthropicSystemMessageContentBlock,
    AnthropicSystemPrompt, AnthropicThinking, AnthropicTool, AnthropicUsage,
    AnthropicUserMessageContent, AnthropicUserMessageContentBlock,
};
use crate::providers::anthropic::extensions;
use std::collections::HashMap;

fn extract_header_value(
    headers: Option<&HashMap<String, String>>,
    keys: &[&str],
) -> Option<String> {
    let headers = headers?;
    for key in keys {
        if let Some((_, value)) = headers
            .iter()
            .find(|(header_name, _)| header_name.eq_ignore_ascii_case(key))
        {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn prompt_cache_control_from_headers(
    headers: Option<&HashMap<String, String>>,
) -> Option<AnthropicCacheControl> {
    let has_cache_key = extract_header_value(
        headers,
        &[
            "x-prompt-cache-key",
            "prompt_cache_key",
            "prompt-cache-key",
            "session_id",
        ],
    )
    .is_some();
    if !has_cache_key {
        return None;
    }

    let ttl = extract_header_value(
        headers,
        &["x-prompt-cache-ttl", "prompt_cache_ttl", "prompt-cache-ttl"],
    );

    Some(AnthropicCacheControl {
        type_: "ephemeral".to_string(),
        ttl,
    })
}

fn system_prompt_with_optional_cache(
    system: String,
    cache_control: Option<&AnthropicCacheControl>,
) -> AnthropicSystemPrompt {
    if let Some(cache_control) = cache_control {
        AnthropicSystemPrompt::Blocks(vec![AnthropicSystemMessageContentBlock::Text {
            text: system,
            cache_control: Some(cache_control.clone()),
        }])
    } else {
        AnthropicSystemPrompt::Text(system)
    }
}

fn apply_prompt_cache_to_first_user_message(
    messages: &mut [AnthropicMessageParam],
    cache_control: &AnthropicCacheControl,
) {
    for message in messages.iter_mut() {
        let AnthropicMessageParam::User { content } = message else {
            continue;
        };

        match content {
            AnthropicUserMessageContent::Text(text) => {
                if text.trim().is_empty() {
                    continue;
                }
                let content_text = std::mem::take(text);
                *content = AnthropicUserMessageContent::Blocks(vec![
                    AnthropicUserMessageContentBlock::Text {
                        text: content_text,
                        cache_control: Some(cache_control.clone()),
                    },
                ]);
                return;
            }
            AnthropicUserMessageContent::Blocks(blocks) => {
                for block in blocks.iter_mut() {
                    let AnthropicUserMessageContentBlock::Text {
                        text,
                        cache_control: block_cache_control,
                    } = block
                    else {
                        continue;
                    };
                    if text.trim().is_empty() {
                        continue;
                    }
                    if block_cache_control.is_none() {
                        *block_cache_control = Some(cache_control.clone());
                    }
                    return;
                }
            }
        }
    }
}

impl From<LanguageModelOptions> for AnthropicOptions {
    fn from(options: LanguageModelOptions) -> Self {
        let mut messages = Vec::new();
        let mut request = AnthropicOptions::builder();
        request.model("");
        let prompt_cache_control = prompt_cache_control_from_headers(options.headers.as_ref());
        let mut has_system_prompt = false;

        // TODO: anthropic max_tokens is required. handle compile
        // time checks if not set in core
        let max_tokens = options.max_output_tokens.unwrap_or(10_000);

        if let Some(system) = options.system
            && !system.is_empty()
        {
            request.system(Some(system_prompt_with_optional_cache(
                system,
                prompt_cache_control.as_ref(),
            )));
            has_system_prompt = true;
        } else {
            request.system(None);
        }

        // convert messages to anthropic messages
        for msg in options.messages {
            match msg.message {
                Message::System(s) => {
                    if !s.content.is_empty() {
                        request.system(Some(system_prompt_with_optional_cache(
                            s.content,
                            prompt_cache_control.as_ref(),
                        )));
                        has_system_prompt = true;
                    }
                }
                Message::User(u) => {
                    messages.push(AnthropicMessageParam::User {
                        content: AnthropicUserMessageContent::Text(u.content),
                    });
                }
                Message::Assistant(a) => match a.content {
                    LanguageModelResponseContentType::Text(text) => {
                        messages.push(AnthropicMessageParam::Assistant {
                            content: vec![AnthropicAssistantMessageParamContent::Text { text }],
                        });
                    }
                    LanguageModelResponseContentType::ToolCall(tool) => {
                        messages.push(AnthropicMessageParam::Assistant {
                            content: vec![AnthropicAssistantMessageParamContent::ToolUse {
                                id: tool.tool.id,
                                input: tool.input,
                                name: tool.tool.name,
                            }],
                        });
                    }
                    LanguageModelResponseContentType::Reasoning {
                        content,
                        extensions,
                    } => {
                        // Retrieve Anthropic-specific signature from extensions
                        let signature = extensions
                            .get::<extensions::AnthropicThinkingMetadata>()
                            .signature
                            .clone()
                            .unwrap_or_else(|| content.clone());

                        messages.push(AnthropicMessageParam::Assistant {
                            content: vec![AnthropicAssistantMessageParamContent::Thinking {
                                thinking: content.clone(),
                                signature,
                            }],
                        });
                    }
                    LanguageModelResponseContentType::NotSupported(_) => {}
                    // Tool approval requests are handled internally by the SDK
                    LanguageModelResponseContentType::ToolApprovalRequest(_) => {}
                },
                Message::Tool(tool) => {
                    messages.push(AnthropicMessageParam::User {
                        content: AnthropicUserMessageContent::Blocks(vec![
                            AnthropicUserMessageContentBlock::ToolResult {
                                tool_use_id: tool.tool.id,
                                content: tool.output.unwrap_or_default().to_string(),
                            },
                        ]),
                    });
                }
                Message::Developer(dev) => {
                    messages.push(AnthropicMessageParam::User {
                        content: AnthropicUserMessageContent::Text(format!(
                            "<developer>\n{}\n</developer>",
                            dev
                        )),
                    });
                }
                // Tool approval messages are handled internally by the SDK
                Message::ToolApproval(_) => {}
            }
        }

        if let Some(cache_control) = prompt_cache_control.as_ref()
            && !has_system_prompt
        {
            apply_prompt_cache_to_first_user_message(&mut messages, cache_control);
        }
        // update messages
        request.messages(messages);

        // convert tools to anthropic tools
        if let Some(tools) = options.tools {
            request.tools(Some(
                tools
                    .tools
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .iter()
                    .map(|t| {
                        let tool = t.clone();
                        let mut tool_schema = tool.input_schema.to_value();
                        if let Some(schema) = tool_schema.as_object_mut() {
                            schema.remove("$schema");
                        };
                        AnthropicTool {
                            name: tool.name,
                            description: tool.description,
                            input_schema: tool_schema,
                        }
                    })
                    .collect(),
            ));
        }

        // convert reasoning to antropic thinking
        request.thinking(options.reasoning_effort.map(|effort| match effort {
            // Low is 25% of the max_tokens
            ReasoningEffort::Low => AnthropicThinking::Enable {
                budget_tokens: (max_tokens / 4) as usize,
            },
            // Medium is 50% of the max_tokens
            ReasoningEffort::Medium => AnthropicThinking::Enable {
                budget_tokens: (max_tokens / 2) as usize,
            },
            // High is 75% of the max_tokens
            ReasoningEffort::High => AnthropicThinking::Enable {
                budget_tokens: (max_tokens - (max_tokens / 4)) as usize,
            },
        }));

        request.build().expect("Failed to build AntropicRequest")
    }
}

impl From<AnthropicUsage> for Usage {
    fn from(usage: AnthropicUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            cached_tokens: Some(usage.cache_creation_input_tokens + usage.cache_read_input_tokens),
            reasoning_tokens: None,
        }
    }
}

impl From<AnthropicMessageDeltaUsage> for Usage {
    fn from(usage: AnthropicMessageDeltaUsage) -> Self {
        Self {
            input_tokens: Some(usage.input_tokens.unwrap_or(0)),
            output_tokens: Some(usage.output_tokens),
            cached_tokens: Some(
                usage.cache_creation_input_tokens.unwrap_or(0)
                    + usage.cache_read_input_tokens.unwrap_or(0),
            ),
            reasoning_tokens: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::messages::UserMessage;

    #[test]
    fn prompt_cache_key_adds_cache_control_to_system_prompt() {
        let options = LanguageModelOptions {
            system: Some("System instructions".to_string()),
            headers: Some(HashMap::from([(
                "x-prompt-cache-key".to_string(),
                "session-abc".to_string(),
            )])),
            ..Default::default()
        };

        let request: AnthropicOptions = options.into();
        let Some(AnthropicSystemPrompt::Blocks(blocks)) = request.system else {
            panic!("expected system blocks");
        };
        let Some(AnthropicSystemMessageContentBlock::Text { cache_control, .. }) = blocks.first()
        else {
            panic!("expected text system block");
        };
        assert_eq!(
            cache_control.as_ref().map(|value| value.type_.as_str()),
            Some("ephemeral")
        );
    }

    #[test]
    fn prompt_cache_key_adds_cache_control_to_first_user_without_system_prompt() {
        let options = LanguageModelOptions {
            messages: vec![Message::User(UserMessage::new("User prompt")).into()],
            headers: Some(HashMap::from([(
                "session_id".to_string(),
                "session-xyz".to_string(),
            )])),
            ..Default::default()
        };

        let request: AnthropicOptions = options.into();
        let Some(AnthropicMessageParam::User { content }) = request.messages.first() else {
            panic!("expected first user message");
        };
        let AnthropicUserMessageContent::Blocks(blocks) = content else {
            panic!("expected user content blocks");
        };
        let Some(AnthropicUserMessageContentBlock::Text { cache_control, .. }) = blocks.first()
        else {
            panic!("expected user text block");
        };
        assert_eq!(
            cache_control.as_ref().map(|value| value.type_.as_str()),
            Some("ephemeral")
        );
    }
}
