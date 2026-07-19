# Roboflow Integration Status

## Status Date

2026-07-19

## Tools Searched For

Tool discovery was queried for Roboflow Workflow capabilities with: `roboflow workflows_get workflows_run workflow specs`.

## Tools Found

The following MCP tools were exposed:

- `mcp__roboflow.workflows_get`
  - Purpose: get details for a saved workflow.
  - Input: `workflow_id` string, the workflow URL slug or ID.
- `mcp__roboflow.workflows_run`
  - Purpose: execute a saved published Workflow.
  - Inputs: `workflow_id`, `images` map, optional `parameters` map.
  - Tool description states images must be public URLs or base64-encoded data; local paths do not work.
- `mcp__roboflow.workflow_specs_run`
  - Purpose: execute an inline workflow specification.
  - Inputs: `specification`, `images`, optional `parameters`.
- `mcp__roboflow.workflows_list`, `workflows_create`, `workflows_update`, and `agent_workflow_publish` were also listed, but live mutation tools are outside this refactor scope.

## Read-Only Workflow Description Captured

`workflows_get` was called for:

`evn-object-detection-vevn-object-detection-cnyo0-2-yolo11n-t1-logic`

The returned saved workflow definition contains:

- Name: `EVN-Object-Detection vevn-object-detection-cnyo0-2-yolo11n-t1 Logic`
- URL slug: `evn-object-detection-vevn-object-detection-cnyo0-2-yolo11n-t1-logic`
- Input: `image` of type `InferenceImage`
- Runtime parameters on wrapper workflow: none
- Step: `roboflow_core/inner_workflow@v1` named `model`
- Output: `predictions` of type `JsonField`, selector `$steps.model.predictions`
- Noted issue: wrapper parameter bindings include `model_id`, which previously caused Roboflow serverless to reject execution because the child workflow did not declare `model_id` as an input.

## Live Run Attempt

A minimal live run was attempted with a base64-encoded local dataset image because Roboflow MCP cannot read local paths.

The action was rejected by the safety layer before execution because it would upload local repository image contents to Roboflow, an external destination not established as trusted internal storage. No workaround was attempted.

## Missing Capabilities or Blockers

- Explicit user approval is required before uploading local repository images to Roboflow for live validation.
- The current published wrapper workflow appears structurally invalid for execution due to its `model_id` child workflow binding.
- The generic adapter intentionally does not claim a Roboflow MCP schema beyond the MCP tool descriptions observed above.

## Code Complete Without MCP

The provider-independent harness is complete and tested offline:

- `edge/harness/`
- `edge/loop/`
- `edge/context/`
- `edge/prompts/`
- `edge/tools/`
- `edge/providers/base.py`
- `edge/providers/roboflow/fake.py`
- `edge/memory/`

The fake provider is deterministic, local-only, and clearly marks results as fake.

## Remaining Adapter Work

- Implement `edge/providers/roboflow/adapter.py` using actual MCP request/response schemas after approved live validation is possible.
- Keep raw MCP payloads inside the adapter.
- Convert `workflows_get` output into internal `WorkflowDescription`.
- Convert `workflows_run` output into internal `WorkflowRunResult`.
- Add contract tests using sanitized captured payload fixtures.
- Add a live smoke test guarded by an explicit environment flag and approval workflow.

## How To Validate Once Approved

1. Confirm the published Roboflow workflow executes in Roboflow UI or fix/publish the wrapper workflow.
2. Confirm explicit approval to upload a selected local fixture, or use a public non-sensitive HTTPS image.
3. Run `workflows_get` and record sanitized schema fixtures.
4. Run `workflows_run` once and record sanitized output fixtures.
5. Implement the adapter mapping to internal provider-neutral models.
6. Run offline contract tests plus the guarded live smoke test.
