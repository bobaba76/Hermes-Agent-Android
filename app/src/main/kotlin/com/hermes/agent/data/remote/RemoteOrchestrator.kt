package com.hermes.agent.data.remote

import com.hermes.agent.domain.agent.ExecutionOrigin
import com.hermes.agent.domain.agent.Orchestrator
import com.hermes.agent.domain.agent.OrchestratorEvent
import com.hermes.agent.domain.llm.LlmMessage
import com.hermes.agent.domain.llm.ToolCall
import com.hermes.agent.domain.model.AgentRole
import com.hermes.agent.domain.model.ExecutionPlan
import com.hermes.agent.domain.model.ExecutionStep
import com.hermes.agent.domain.model.StepStatus
import com.hermes.agent.domain.tool.ToolConfirmationService
import com.hermes.agent.util.DispatcherProvider
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import timber.log.Timber
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Thin-client [Orchestrator] that delegates all agent execution to a PC
 * Hermes gateway over the Runs API.
 *
 * The PC is the canonical agent runtime and conversation store. This class:
 *   1. Starts a remote run via [GatewayApiClient.startRun], passing the
 *      conversation ID as `session_id` so the gateway loads the session's
 *      transcript (seamless handoff).
 *   2. Streams lifecycle events from `GET /v1/runs/{id}/events`.
 *   3. Maps each [GatewayEvent] to an [OrchestratorEvent] the existing chat
 *      UI already renders.
 *   4. When the gateway emits an `approval.request`, surfaces the existing
 *      phone-side approval dialog via [ToolConfirmationService] — the same
 *      mechanism [OrchestratorImpl] uses — and forwards the user's decision
 *      back to the PC via `POST /v1/runs/{id}/approval`.
 *
 * No local agent execution, no local LLM call, no local tool execution. The
 * phone is a display + approval surface; the PC does the work.
 */
@Singleton
class RemoteOrchestrator @Inject constructor(
    private val gatewayClient: GatewayApiClient,
    private val toolConfirmationService: ToolConfirmationService,
    private val json: Json,
    private val dispatchers: DispatcherProvider,
) : Orchestrator {

    override fun run(
        conversationId: String,
        userMessage: String,
        recentMessages: List<LlmMessage>,
        origin: ExecutionOrigin,
    ): Flow<OrchestratorEvent> = flow {
        // recentMessages is intentionally ignored: the PC gateway loads the
        // session transcript from the session_id we pass to startRun.
        try {
            val runId = gatewayClient.startRun(userMessage, sessionId = conversationId)
            Timber.tag("RemoteOrchestrator").i("Started remote run=%s session=%s", runId, conversationId)

            // Emit a minimal plan so the UI's plan indicator lights up. The
            // real plan lives on the PC; this is a single-step placeholder
            // that satisfies the existing UI contract.
            val stepId = UUID.randomUUID().toString()
            val plan = ExecutionPlan(
                id = UUID.randomUUID().toString(),
                conversationId = conversationId,
                userMessage = userMessage,
                steps = listOf(
                    ExecutionStep(
                        id = stepId,
                        agentRole = AgentRole.DEFAULT,
                        description = "Remote gateway run",
                        status = StepStatus.RUNNING,
                        startedAt = System.currentTimeMillis(),
                    ),
                ),
                createdAt = System.currentTimeMillis(),
            )
            emit(OrchestratorEvent.PlanReady(plan))
            emit(OrchestratorEvent.StepStarted(stepId, AgentRole.DEFAULT))

            gatewayClient.streamRunEvents(runId).collect { event ->
                when (event) {
                    is GatewayEvent.MessageDelta -> {
                        emit(OrchestratorEvent.ReplyToken(event.text))
                    }

                    is GatewayEvent.MessageComplete -> {
                        emit(OrchestratorEvent.ReplyComplete(
                            event.text,
                            AgentRole.DEFAULT,
                            isOnDevice = false,
                        ))
                    }

                    is GatewayEvent.ToolStarted -> {
                        val call = parseToolCall(event.callId, event.name, event.arguments)
                        emit(OrchestratorEvent.ToolCallRequested(call, requiresConfirmation = false))
                    }

                    is GatewayEvent.ToolCompleted -> {
                        val call = parseToolCall(event.callId, event.name, "")
                        emit(OrchestratorEvent.ToolCallResult(call, event.output, event.success))
                    }

                    is GatewayEvent.ApprovalRequested -> {
                        // Surface the phone-side approval dialog via the existing
                        // ToolConfirmationService — the same mechanism
                        // OrchestratorImpl uses. The service suspends on a
                        // CompletableDeferred; the UI observes pendingRequest
                        // and calls submitConfirmation() when the user taps.
                        val call = parseToolCall(event.callId, event.toolName, event.arguments)
                        emit(OrchestratorEvent.ToolCallRequested(call, requiresConfirmation = true))
                        val approved = toolConfirmationService.awaitConfirmation(call)
                        Timber.tag("RemoteOrchestrator").i(
                            "Approval call=%s approved=%s — forwarding to PC",
                            event.toolName, approved,
                        )
                        gatewayClient.submitApproval(runId, approved)
                    }

                    is GatewayEvent.RunCompleted -> {
                        // If no MessageComplete was emitted, emit a final reply
                        // from the run output so the UI still gets a terminal event.
                        emit(OrchestratorEvent.ReplyComplete(
                            event.output,
                            AgentRole.DEFAULT,
                            isOnDevice = false,
                        ))
                    }

                    is GatewayEvent.RunFailed -> {
                        emit(OrchestratorEvent.Failed(event.message))
                    }

                    is GatewayEvent.SubagentStarted -> {
                        Timber.tag("RemoteOrchestrator").d(
                            "Subagent started: child=%s delegation=%s",
                            event.childSessionId, event.delegationId,
                        )
                    }

                    is GatewayEvent.SubagentCompleted -> {
                        Timber.tag("RemoteOrchestrator").d(
                            "Subagent completed: child=%s status=%s",
                            event.childSessionId, event.status,
                        )
                    }

                    is GatewayEvent.Unknown -> {
                        Timber.tag("RemoteOrchestrator").d("Unknown event: %s", event.type)
                    }
                }
            }

            emit(OrchestratorEvent.StepFinished(stepId, success = true))
        } catch (e: Exception) {
            Timber.tag("RemoteOrchestrator").e(e, "Remote run failed")
            emit(OrchestratorEvent.Failed(e.message ?: "Remote gateway error"))
        }
    }.flowOn(dispatchers.io)

    /**
     * Parse a tool call from the gateway's event payload.
     *
     * The gateway sends `arguments` as a JSON string (the same shape the LLM
     * produced). We parse it into a `Map<String, JsonElement>` so the
     * existing [ToolCall] domain model can carry it unchanged.
     */
    private fun parseToolCall(callId: String, name: String, argumentsJson: String): ToolCall {
        val arguments: Map<String, kotlinx.serialization.json.JsonElement> = if (argumentsJson.isBlank()) {
            emptyMap()
        } else {
            runCatching {
                json.parseToJsonElement(argumentsJson) as? JsonObject
            }.getOrNull()?.toMap() ?: emptyMap()
        }
        return ToolCall(
            id = callId.ifBlank { UUID.randomUUID().toString() },
            name = name,
            arguments = arguments,
        )
    }
}
