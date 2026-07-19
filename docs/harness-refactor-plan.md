# Harness Refactor Plan

## Proposed Directory Tree

```text
edge/
    __init__.py
    harness/
        __init__.py
        runtime.py
        models.py
        errors.py
        events.py
        retry.py
        checkpoint.py
        logging.py
    loop/
        __init__.py
        controller.py
        state.py
        planner.py
        actions.py
        verifier.py
        policies.py
    context/
        __init__.py
        assembler.py
        models.py
        repository_context.py
        history.py
        retrieval.py
        capabilities.py
    prompts/
        __init__.py
        builder.py
        templates.py
        schemas.py
    tools/
        __init__.py
        base.py
        registry.py
        executor.py
        results.py
        policies.py
    providers/
        __init__.py
        base.py
        roboflow/
            __init__.py
            interface.py
            adapter.py
            models.py
            errors.py
            fake.py
    memory/
        __init__.py
        store.py
        models.py
        checkpoint_store.py
        in_memory.py
        file_store.py
```

## Mapping From Current Modules

- `server_pc/app/roboflow_detector.py` remains a server compatibility detector. Later it should delegate to `edge.providers.roboflow.adapter` through the provider interface.
- `shared/services/roboflow_workflow_client.py` is provider-specific implementation detail. Later it should move behind `edge/providers/roboflow/adapter.py` or be wrapped there.
- `shared/services/media_downloader.py` remains reusable for service/runtime URL downloads. The local harness vertical slice starts with local file paths only.
- `shared/services/callback_service.py` remains a service-level callback client. Later expose it as a typed tool with side-effect metadata.
- `shared/messaging/rabbitmq_client.py` remains service infrastructure. Later queue consumption should create harness triggers rather than running inference inline.
- `edge_raspberry/app/detector.py` and `server_pc/app/detector.py` remain legacy detector implementations. Later wrap each as providers/tools.
- `inference/gateway.py` should not be preserved as a harness design pattern; it should be decomposed if revived.

## Public Interfaces

- `VisionWorkflowProvider.describe_workflow(workflow_ref) -> WorkflowDescription`
- `VisionWorkflowProvider.run_workflow(request) -> WorkflowRunResult`
- `Tool.execute(input) -> ToolResult`
- `ToolExecutor.execute(tool_name, action_id, input) -> ToolResult`
- `ContextAssembler.assemble(run_state, goal, capabilities, budget) -> ExecutionContext`
- `PromptBuilder.build_action_prompt(context) -> PromptMessageBundle`
- `HarnessRuntime.start_run(trigger) -> RunResult`
- `HarnessRuntime.resume_run(run_id) -> RunResult`
- `CheckpointStore.save(snapshot)` and `CheckpointStore.load_latest(run_id)`

## Migration Sequence

1. Add provider-independent dataclasses/enums for states, errors, tools, providers, checkpoints, events, and context.
2. Implement fake vision provider and vision workflow tool.
3. Implement local-image trigger validation.
4. Implement explicit loop state transitions and a controller for one offline vision-workflow action.
5. Add file checkpoint store with redaction.
6. Add structured event logger.
7. Add tests for success/failure/retry/resume/redaction/state validity.
8. Add Roboflow adapter skeleton with TODOs and capability-gap docs if MCP or live schemas are unavailable.
9. Later migrate server/edge consumers to create harness triggers and execute through `HarnessRuntime`.

## Compatibility Approach

- Do not rewrite `edge_raspberry`, `server_pc`, or `shared` service behavior in this vertical slice.
- Keep current Pydantic DTOs and detector protocol intact.
- Add wrappers/adapters rather than moving existing code immediately.
- The new `edge` package is generic harness infrastructure, distinct from the existing `edge_raspberry` runtime.

## Testing Strategy

Offline unit tests will cover:

- Successful local-image fake-provider run.
- Missing image and unsupported image type validation.
- Retryable tool failure followed by success.
- Retry exhaustion.
- Capability unavailable.
- Invalid provider response.
- Verification failure.
- Checkpoint creation and resume from checkpoint.
- Secret redaction in logs/checkpoints.
- State-transition validity.
- Duplicate execution prevention for non-idempotent actions.

No test will require internet, Roboflow MCP, a Roboflow account, RabbitMQ, local model weights, or backend services.

## Risks

- The name `edge/` may be confused with `edge_raspberry/`; docs must clarify that `edge/` is harness infrastructure.
- Existing tests have discovery quirks because some test files are not import-safe operational scripts.
- The current Roboflow workflow is known to fail live until republished/fixed.
- Server integration from earlier work still needs Docker env wiring for `ROBOFLOW_API_KEY` before production use.

## Deferred Roboflow Work

- Inspect actual Roboflow MCP tool schemas when available.
- Keep raw MCP request/response payloads inside `edge/providers/roboflow/adapter.py`.
- Convert real Roboflow workflow descriptions/results into internal `WorkflowDescription` and `WorkflowRunResult` models.
- Add adapter contract tests based on captured real schemas.
- Run a minimal workflow with a representative local image after the published workflow is fixed.

## Definition of Done

- Audit and plan docs exist.
- Provider-independent vertical slice runs locally on an image file.
- Fake provider is deterministic and clearly marked fake.
- Checkpoints and structured logs are produced without secrets or binary payloads.
- Tests cover the required success and failure cases offline.
- Roboflow integration status is documented without fabricating live success.
