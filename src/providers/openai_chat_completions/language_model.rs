//! Language model implementation for the OpenAI Chat Completions provider.

use crate::core::capabilities::ModelName;
use crate::core::client::Client;
use crate::core::language_model::{
    LanguageModel, LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream,
};
use crate::core::messages::AssistantMessage;
use crate::core::tools::ToolCallInfo;
use crate::error::Result;
use crate::providers::openai_chat_completions::OpenAIChatCompletions;
use crate::providers::openai_chat_completions::client::{self, types};
use async_trait::async_trait;
use futures::StreamExt;
use std::collections::HashMap;

#[async_trait]
impl<M: ModelName> LanguageModel for OpenAIChatCompletions<M> {
    fn name(&self) -> String {
        self.options.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        // Extract additional headers before converting options
        let additional_headers = options.headers_as_header_map();
        let additional_headers = if additional_headers.is_empty() {
            None
        } else {
            Some(additional_headers)
        };

        let mut chat_options: client::ChatCompletionsOptions = options.into();
        chat_options.model = self.options.model.clone();
        apply_openrouter_anthropic_prompt_caching(&self.settings.provider_name, &mut chat_options);
        self.options = chat_options;

        let response: types::ChatCompletionsResponse = self
            .send(&self.settings.base_url, additional_headers)
            .await?;

        // Convert choices to LanguageModelResponse
        let mut contents = Vec::new();

        for choice in response.choices {
            // Handle text content
            if let Some(text) = extract_text_content(choice.message.content)
                && !text.is_empty()
            {
                contents.push(LanguageModelResponseContentType::Text(text));
            }

            // Handle tool calls
            if let Some(tool_calls) = choice.message.tool_calls {
                for tool_call in tool_calls {
                    let mut tool_info = ToolCallInfo::new(tool_call.function.name);
                    tool_info.id(tool_call.id);
                    tool_info.input(
                        serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
                    );
                    contents.push(LanguageModelResponseContentType::ToolCall(tool_info));
                }
            }
        }

        Ok(LanguageModelResponse {
            contents,
            usage: response.usage.map(|u| u.into()),
        })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        // Extract additional headers before converting options
        let additional_headers = options.headers_as_header_map();
        let additional_headers = if additional_headers.is_empty() {
            None
        } else {
            Some(additional_headers)
        };

        let mut chat_options: client::ChatCompletionsOptions = options.into();
        chat_options.model = self.options.model.clone();
        chat_options.stream = Some(true);
        chat_options.stream_options = Some(types::StreamOptions {
            include_usage: Some(true),
            include_obfuscation: Some(false),
        });
        apply_openrouter_anthropic_prompt_caching(&self.settings.provider_name, &mut chat_options);
        self.options = chat_options;

        let stream = self
            .send_and_stream(&self.settings.base_url, additional_headers)
            .await?;

        // State for accumulating tool calls across chunks
        let mut accumulated_tool_calls: HashMap<u32, (String, String, String)> = HashMap::new();

        // Map stream events to SDK stream chunks
        let stream = stream.map(move |evt_res| match evt_res {
            Ok(types::ChatCompletionsStreamEvent::Chunk(chunk)) => {
                let mut results = Vec::new();

                for choice in chunk.choices {
                    // Text delta
                    if let Some(content) = choice.delta.content
                        && !content.is_empty()
                    {
                        results.push(LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Text(content),
                        ));
                    }

                    // Accumulate tool call deltas
                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for tool_call in tool_calls {
                            let entry = accumulated_tool_calls.entry(tool_call.index).or_insert((
                                String::new(),
                                String::new(),
                                String::new(),
                            ));

                            // Accumulate ID
                            if let Some(id) = tool_call.id {
                                entry.0 = id;
                            }

                            // Accumulate name and arguments
                            if let Some(function) = tool_call.function {
                                if let Some(name) = function.name {
                                    entry.1 = name;
                                }
                                if let Some(args) = function.arguments {
                                    entry.2.push_str(&args);
                                    results.push(LanguageModelStreamChunk::Delta(
                                        LanguageModelStreamChunkType::ToolCall(args),
                                    ));
                                }
                            }
                        }
                    }

                    if let Some(finish_reason) = choice.finish_reason {
                        let usage = chunk.usage.clone().map(|u| u.into());

                        match finish_reason.as_str() {
                            "stop" | "length" => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                            }
                            "tool_calls" | "function_call" => {
                                // Send accumulated tool calls
                                for (id, name, args) in accumulated_tool_calls.values() {
                                    let mut tool_info = ToolCallInfo::new(name.clone());
                                    tool_info.id(id.clone());
                                    tool_info.input(serde_json::from_str(args).unwrap_or_else(
                                        |_| serde_json::Value::Object(serde_json::Map::new()),
                                    ));
                                    results.push(LanguageModelStreamChunk::Done(
                                        AssistantMessage {
                                            content: LanguageModelResponseContentType::ToolCall(
                                                tool_info,
                                            ),
                                            usage: usage.clone(),
                                        },
                                    ));
                                }
                            }
                            "content_filter" => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                                results.push(LanguageModelStreamChunk::Delta(
                                    LanguageModelStreamChunkType::Failed(
                                        "Content filtered".to_string(),
                                    ),
                                ));
                            }
                            // For any unknown finish reason, treat as normal completion
                            _ => {
                                results.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Text(String::new()),
                                    usage,
                                }));
                            }
                        }
                    }
                }

                Ok(results)
            }
            Ok(types::ChatCompletionsStreamEvent::Open) => Ok(vec![]),
            Ok(types::ChatCompletionsStreamEvent::Done) => Ok(vec![]),
            Ok(types::ChatCompletionsStreamEvent::Error(e)) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Failed(e),
                )])
            }
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

fn is_openrouter_anthropic_model(provider_name: &str, model_name: &str) -> bool {
    provider_name.eq_ignore_ascii_case("openrouter") && model_name.starts_with("anthropic/")
}

fn apply_openrouter_anthropic_prompt_caching(
    provider_name: &str,
    options: &mut client::ChatCompletionsOptions,
) {
    if !is_openrouter_anthropic_model(provider_name, &options.model) {
        return;
    }

    let target_index = options
        .messages
        .iter()
        .position(|msg| matches!(msg.role, types::Role::System))
        .or_else(|| {
            options
                .messages
                .iter()
                .position(|msg| matches!(msg.role, types::Role::User))
        });

    let Some(target_index) = target_index else {
        return;
    };

    apply_ephemeral_cache_control(&mut options.messages[target_index]);
}

fn apply_ephemeral_cache_control(message: &mut types::ChatMessage) {
    let cache_control = types::CacheControl {
        type_: "ephemeral".to_string(),
        ttl: None,
    };

    match message.content.as_mut() {
        Some(types::ChatMessageContent::Text(text)) => {
            if text.trim().is_empty() {
                return;
            }
            let part = types::ChatMessageContentPart::Text(types::ChatMessageContentPartText {
                text: std::mem::take(text),
                cache_control: Some(cache_control),
                extra: HashMap::new(),
            });
            message.content = Some(types::ChatMessageContent::Parts(vec![part]));
        }
        Some(types::ChatMessageContent::Parts(parts)) => {
            if has_cache_control(parts) {
                return;
            }
            for part in parts.iter_mut() {
                if let types::ChatMessageContentPart::Text(text_part) = part {
                    if text_part.text.trim().is_empty() {
                        continue;
                    }
                    text_part.cache_control = Some(cache_control.clone());
                    break;
                }
            }
        }
        None => {}
    }
}

fn has_cache_control(parts: &[types::ChatMessageContentPart]) -> bool {
    parts.iter().any(|part| match part {
        types::ChatMessageContentPart::Text(text) => text.cache_control.is_some(),
        types::ChatMessageContentPart::ImageUrl(image) => image.cache_control.is_some(),
        types::ChatMessageContentPart::InputAudio(audio) => audio.cache_control.is_some(),
        types::ChatMessageContentPart::File(file) => file.cache_control.is_some(),
    })
}

fn extract_text_content(content: Option<types::ChatMessageContent>) -> Option<String> {
    match content {
        Some(types::ChatMessageContent::Text(text)) => Some(text),
        Some(types::ChatMessageContent::Parts(parts)) => {
            let mut out = String::new();
            for part in parts {
                if let types::ChatMessageContentPart::Text(text_part) = part {
                    out.push_str(&text_part.text);
                }
            }
            if out.is_empty() { None } else { Some(out) }
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_anthropic_adds_cache_control_to_system_prompt() {
        let mut options = client::ChatCompletionsOptions {
            model: "anthropic/claude-sonnet-4.5".to_string(),
            messages: vec![types::ChatMessage {
                role: types::Role::System,
                content: Some(types::ChatMessageContent::text("System prompt")),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            ..Default::default()
        };

        apply_openrouter_anthropic_prompt_caching("OpenRouter", &mut options);

        let content = options
            .messages
            .first()
            .and_then(|m| m.content.clone())
            .expect("message content");
        let types::ChatMessageContent::Parts(parts) = content else {
            panic!("expected content parts");
        };
        let Some(types::ChatMessageContentPart::Text(text_part)) = parts.first() else {
            panic!("expected text part");
        };
        assert_eq!(
            text_part.cache_control.as_ref().map(|c| c.type_.as_str()),
            Some("ephemeral")
        );
    }

    #[test]
    fn non_openrouter_provider_does_not_modify_messages() {
        let original = types::ChatMessage {
            role: types::Role::System,
            content: Some(types::ChatMessageContent::text("System prompt")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let mut options = client::ChatCompletionsOptions {
            model: "anthropic/claude-sonnet-4.5".to_string(),
            messages: vec![original.clone()],
            ..Default::default()
        };

        apply_openrouter_anthropic_prompt_caching("openai-chat", &mut options);

        assert_eq!(options.messages[0], original);
    }
}
