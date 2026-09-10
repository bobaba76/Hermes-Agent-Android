package com.hermes.agent.data.remote

import com.hermes.agent.domain.settings.SettingsRepository
import com.hermes.agent.util.DispatcherProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import timber.log.Timber
import java.io.BufferedReader
import java.io.IOException
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext

/**
 * HTTP + SSE client for the PC Hermes gateway's REST API.
 *
 * Wraps the Runs API (for [RemoteOrchestrator]) and the Sessions API (for
 * [RemoteConversationRepository]). All calls send `Authorization: Bearer
 * <key>` and read the base URL + key from [SettingsRepository] at call time
 * so settings changes take effect without a restart of this client.
 *
 * The gateway's API surface is documented in the NousResearch/hermes-agent
 * `gateway/platforms/api_server.py` module:
 *
 *   - `POST /v1/runs` — start a run, returns `run_id` (202)
 *   - `GET  /v1/runs/{id}/events` — SSE stream of lifecycle events
 *   - `POST /v1/runs/{id}/approval` — resolve a pending approval
 *   - `POST /v1/runs/{id}/stop` — interrupt a running agent
 *   - `GET  /api/sessions` — list sessions
 *   - `POST /api/sessions` — create a session
 *   - `GET  /api/sessions/{id}` — read session metadata
 *   - `PATCH /api/sessions/{id}` — update title
 *   - `DELETE /api/sessions/{id}` — delete a session
 *   - `GET  /api/sessions/{id}/messages` — message history
 *   - `POST /api/sessions/{id}/fork` — branch a session
 */
@Singleton
class GatewayApiClient @Inject constructor(
    private val client: OkHttpClient,
    private val json: Json,
    private val settingsRepository: SettingsRepository,
    private val dispatchers: DispatcherProvider,
) {

    private val jsonMediaType = "application/json".toMediaType()

    private suspend fun baseUrl(): String =
        settingsRepository.current().remoteGatewayUrl.trimEnd('/')

    private suspend fun apiKey(): String =
        settingsRepository.current().remoteGatewayApiKey

    private fun authBuilder(url: String, key: String): Request.Builder =
        Request.Builder().url(url).header("Authorization", "Bearer $key")

    // ── Runs API ──────────────────────────────────────────────────────────

    /**
     * Start a new agent run on the PC gateway. Returns the `run_id`.
     *
     * When [sessionId] is provided, the gateway loads that session's active
     * transcript so the run has full conversation context — this is what
     * enables seamless handoff (the PC is the canonical store).
     */
    suspend fun startRun(input: String, sessionId: String?): String = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val body = buildString {
            append("{")
            append("\"input\":").append(json.encodeToString(JsonObject.serializer(), JsonObject(mapOf("content" to JsonPrimitive(input)))))
            if (sessionId != null) {
                append(",\"session_id\":").append(json.encodeToString(JsonPrimitive.serializer(), JsonPrimitive(sessionId)))
            }
            append("}")
        }
        val request = authBuilder("$base/v1/runs", key)
            .post(body.toRequestBody(jsonMediaType))
            .build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) throw IOException("startRun failed: ${response.code}")
        val responseBody = response.body?.string().orEmpty()
        response.close()
        val parsed = json.parseToJsonElement(responseBody).jsonObject
        parsed["run_id"]?.jsonPrimitive?.contentOrNull
            ?: throw IOException("startRun: no run_id in response")
    }

    /**
     * Stream lifecycle events for a run as a Kotlin [Flow].
     *
     * Cancelling the flow cancels the OkHttp call. Each `data: {...}` SSE
     * line is parsed into a [GatewayEvent]. The flow completes when the
     * stream ends (run completed/failed/cancelled) or the connection drops.
     */
    fun streamRunEvents(runId: String): Flow<GatewayEvent> = flow {
        val base = baseUrl()
        val key = apiKey()
        val request = authBuilder("$base/v1/runs/$runId/events", key).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            response.close()
            emit(GatewayEvent.RunFailed("events stream failed: ${response.code}"))
            return@flow
        }
        val body = response.body ?: run {
            response.close()
            emit(GatewayEvent.RunFailed("events stream: empty body"))
            return@flow
        }
        val reader = BufferedReader(body.charStream())
        try {
            var eventType = ""
            val dataBuilder = StringBuilder()
            while (coroutineContext[Job]?.isActive != false) {
                val line = reader.readLine() ?: break
                when {
                    line.startsWith("event:") -> eventType = line.removePrefix("event:").trim()
                    line.startsWith("data:") -> {
                        dataBuilder.append(line.removePrefix("data:").trim())
                    }
                    line.isEmpty() && dataBuilder.isNotEmpty() -> {
                        val data = dataBuilder.toString()
                        dataBuilder.setLength(0)
                        parseEvent(eventType.ifEmpty { "message" }, data)?.let { emit(it) }
                        eventType = ""
                    }
                }
            }
        } catch (e: IOException) {
            if (coroutineContext[Job]?.isActive != false) {
                Timber.tag("GatewayClient").w(e, "SSE stream interrupted")
            }
        } finally {
            response.close()
        }
    }.flowOn(dispatchers.io)

    /** Resolve a pending approval for a run. */
    suspend fun submitApproval(runId: String, approved: Boolean) = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val body = """{"approved":$approved}"""
        val request = authBuilder("$base/v1/runs/$runId/approval", key)
            .post(body.toRequestBody(jsonMediaType))
            .build()
        val response = client.newCall(request).execute()
        response.close()
        if (!response.isSuccessful) {
            Timber.tag("GatewayClient").w("submitApproval failed: %d", response.code)
        }
    }

    /** Interrupt a running agent. */
    suspend fun stopRun(runId: String) = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val request = authBuilder("$base/v1/runs/$runId/stop", key)
            .post("".toRequestBody(jsonMediaType))
            .build()
        val response = client.newCall(request).execute()
        response.close()
    }

    // ── Sessions API ──────────────────────────────────────────────────────

    /** List all sessions on the PC gateway. */
    suspend fun listSessions(): List<RemoteSession> = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val request = authBuilder("$base/api/sessions", key).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            response.close()
            throw IOException("listSessions failed: ${response.code}")
        }
        val body = response.body?.string().orEmpty()
        response.close()
        parseSessions(body)
    }

    /** Create a new empty session on the PC gateway. */
    suspend fun createSession(title: String): RemoteSession = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val body = """{"title":${json.encodeToString(JsonPrimitive.serializer(), JsonPrimitive(title))}}"""
        val request = authBuilder("$base/api/sessions", key)
            .post(body.toRequestBody(jsonMediaType))
            .build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            response.close()
            throw IOException("createSession failed: ${response.code}")
        }
        val responseBody = response.body?.string().orEmpty()
        response.close()
        parseSession(responseBody)
    }

    /** Read a session's metadata. */
    suspend fun getSession(id: String): RemoteSession = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val request = authBuilder("$base/api/sessions/$id", key).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            response.close()
            throw IOException("getSession failed: ${response.code}")
        }
        val body = response.body?.string().orEmpty()
        response.close()
        parseSession(body)
    }

    /** Update a session's title. */
    suspend fun updateSession(id: String, title: String) = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val body = """{"title":${json.encodeToString(JsonPrimitive.serializer(), JsonPrimitive(title))}}"""
        val request = authBuilder("$base/api/sessions/$id", key)
            .patch(body.toRequestBody(jsonMediaType))
            .build()
        val response = client.newCall(request).execute()
        response.close()
    }

    /** Delete a session. */
    suspend fun deleteSession(id: String) = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val request = authBuilder("$base/api/sessions/$id", key).delete().build()
        val response = client.newCall(request).execute()
        response.close()
    }

    /** Read a session's message history. */
    suspend fun getSessionMessages(id: String): List<RemoteMessage> = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val request = authBuilder("$base/api/sessions/$id/messages", key).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            response.close()
            throw IOException("getSessionMessages failed: ${response.code}")
        }
        val body = response.body?.string().orEmpty()
        response.close()
        parseMessages(body)
    }

    /** Fork a session into a new branched session. */
    suspend fun forkSession(id: String, title: String): RemoteSession = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val body = """{"title":${json.encodeToString(JsonPrimitive.serializer(), JsonPrimitive(title))}}"""
        val request = authBuilder("$base/api/sessions/$id/fork", key)
            .post(body.toRequestBody(jsonMediaType))
            .build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            response.close()
            throw IOException("forkSession failed: ${response.code}")
        }
        val responseBody = response.body?.string().orEmpty()
        response.close()
        parseSession(responseBody)
    }

    // ── Health check ───────────────────────────────────────────────────────

    /** Ping the gateway's health endpoint. Returns true if reachable. */
    suspend fun healthCheck(): Boolean = withContext(dispatchers.io) {
        val base = baseUrl()
        val key = apiKey()
        val request = authBuilder("$base/health", key).build()
        val response = client.newCall(request).execute()
        val ok = response.isSuccessful
        response.close()
        ok
    }

    // ── Parsing ───────────────────────────────────────────────────────────

    private fun parseEvent(type: String, data: String): GatewayEvent? {
        if (data.isBlank() || data == "[DONE]") return null
        val obj = runCatching { json.parseToJsonElement(data).jsonObject }.getOrElse {
            Timber.tag("GatewayClient").w(it, "could not parse SSE event: %s", data.take(200))
            return null
        }
        return when (type) {
            "message.delta", "assistant.delta" -> {
                val text = obj["delta"]?.jsonPrimitive?.contentOrNull
                    ?: obj["text"]?.jsonPrimitive?.contentOrNull
                    ?: obj["content"]?.jsonPrimitive?.contentOrNull
                    ?: ""
                GatewayEvent.MessageDelta(text)
            }
            "message.complete", "assistant.complete" -> {
                val text = obj["text"]?.jsonPrimitive?.contentOrNull
                    ?: obj["content"]?.jsonPrimitive?.contentOrNull
                    ?: obj["output"]?.jsonPrimitive?.contentOrNull
                    ?: ""
                GatewayEvent.MessageComplete(text)
            }
            "tool.started", "tool.start" -> {
                GatewayEvent.ToolStarted(
                    callId = obj["call_id"]?.jsonPrimitive?.contentOrNull ?: "",
                    name = obj["name"]?.jsonPrimitive?.contentOrNull ?: "",
                    arguments = obj["arguments"]?.jsonPrimitive?.contentOrNull ?: "",
                )
            }
            "tool.completed", "tool.complete" -> {
                GatewayEvent.ToolCompleted(
                    callId = obj["call_id"]?.jsonPrimitive?.contentOrNull ?: "",
                    name = obj["name"]?.jsonPrimitive?.contentOrNull ?: "",
                    output = obj["output"]?.jsonPrimitive?.contentOrNull ?: "",
                    success = obj["success"]?.jsonPrimitive?.contentOrNull?.toBooleanStrictOrNull() ?: true,
                )
            }
            "approval.request" -> {
                GatewayEvent.ApprovalRequested(
                    callId = obj["call_id"]?.jsonPrimitive?.contentOrNull
                        ?: obj["request_id"]?.jsonPrimitive?.contentOrNull ?: "",
                    toolName = obj["tool_name"]?.jsonPrimitive?.contentOrNull
                        ?: obj["name"]?.jsonPrimitive?.contentOrNull ?: "",
                    arguments = obj["arguments"]?.jsonPrimitive?.contentOrNull ?: "",
                )
            }
            "run.completed" -> {
                GatewayEvent.RunCompleted(
                    obj["output"]?.jsonPrimitive?.contentOrNull ?: "",
                )
            }
            "run.failed" -> {
                GatewayEvent.RunFailed(
                    obj["message"]?.jsonPrimitive?.contentOrNull
                        ?: obj["error"]?.jsonPrimitive?.contentOrNull ?: "run failed",
                )
            }
            "subagent.start" -> {
                GatewayEvent.SubagentStarted(
                    childSessionId = obj["child_session_id"]?.jsonPrimitive?.contentOrNull ?: "",
                    delegationId = obj["delegation_id"]?.jsonPrimitive?.contentOrNull ?: "",
                )
            }
            "subagent.complete" -> {
                GatewayEvent.SubagentCompleted(
                    childSessionId = obj["child_session_id"]?.jsonPrimitive?.contentOrNull ?: "",
                    status = obj["status"]?.jsonPrimitive?.contentOrNull ?: "",
                    summary = obj["summary"]?.jsonPrimitive?.contentOrNull ?: "",
                )
            }
            else -> GatewayEvent.Unknown(type, data)
        }
    }

    private fun parseSessions(body: String): List<RemoteSession> {
        val arr = runCatching { json.parseToJsonElement(body) }.getOrElse { return emptyList() }
        return when {
            arr is kotlinx.serialization.json.JsonArray -> arr.mapNotNull { parseSession(it.jsonObject) }
            arr is JsonObject && arr["sessions"] is kotlinx.serialization.json.JsonArray ->
                arr["sessions"]!!.let { it as kotlinx.serialization.json.JsonArray }.mapNotNull { el -> parseSession(el.jsonObject) }
            arr is JsonObject && arr["data"] is kotlinx.serialization.json.JsonArray ->
                arr["data"]!!.let { it as kotlinx.serialization.json.JsonArray }.mapNotNull { el -> parseSession(el.jsonObject) }
            else -> emptyList()
        }
    }

    private fun parseSession(body: String): RemoteSession =
        parseSession(json.parseToJsonElement(body).jsonObject)
            ?: throw IOException("could not parse session: ${body.take(200)}")

    private fun parseSession(obj: JsonObject): RemoteSession? {
        val id = obj["id"]?.jsonPrimitive?.contentOrNull ?: obj["session_id"]?.jsonPrimitive?.contentOrNull ?: return null
        return RemoteSession(
            id = id,
            title = obj["title"]?.jsonPrimitive?.contentOrNull ?: "Untitled",
            createdAt = obj["created_at"]?.jsonPrimitive?.contentOrNull?.toLongOrNull()
                ?: obj["created"]?.jsonPrimitive?.contentOrNull?.toLongOrNull()
                ?: System.currentTimeMillis(),
            updatedAt = obj["updated_at"]?.jsonPrimitive?.contentOrNull?.toLongOrNull()
                ?: obj["updated"]?.jsonPrimitive?.contentOrNull?.toLongOrNull()
                ?: System.currentTimeMillis(),
            messageCount = obj["message_count"]?.jsonPrimitive?.contentOrNull?.toIntOrNull() ?: 0,
        )
    }

    private fun parseMessages(body: String): List<RemoteMessage> {
        val arr = runCatching { json.parseToJsonElement(body) }.getOrElse { return emptyList() }
        val elements = when {
            arr is kotlinx.serialization.json.JsonArray -> arr
            arr is JsonObject && arr["messages"] is kotlinx.serialization.json.JsonArray -> arr["messages"] as kotlinx.serialization.json.JsonArray
            arr is JsonObject && arr["data"] is kotlinx.serialization.json.JsonArray -> arr["data"] as kotlinx.serialization.json.JsonArray
            else -> return emptyList()
        }
        return elements.mapNotNull { el -> parseMessage(el.jsonObject) }
    }

    private fun parseMessage(obj: JsonObject): RemoteMessage? {
        val role = obj["role"]?.jsonPrimitive?.contentOrNull ?: return null
        val content = obj["content"]?.jsonPrimitive?.contentOrNull ?: ""
        return RemoteMessage(
            id = obj["id"]?.jsonPrimitive?.contentOrNull ?: java.util.UUID.randomUUID().toString(),
            role = role,
            content = content,
            timestamp = obj["timestamp"]?.jsonPrimitive?.contentOrNull?.toLongOrNull()
                ?: obj["created_at"]?.jsonPrimitive?.contentOrNull?.toLongOrNull()
                ?: System.currentTimeMillis(),
        )
    }
}

/** A session on the PC gateway, mapped from the `/api/sessions` response. */
data class RemoteSession(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val messageCount: Int,
)

/** A message in a PC gateway session, mapped from `/api/sessions/{id}/messages`. */
data class RemoteMessage(
    val id: String,
    val role: String,
    val content: String,
    val timestamp: Long,
)
