import assert from "node:assert/strict";
import test from "node:test";
import { WebSocket } from "ws";
import { HEARTBEAT_STALE_THRESHOLD_MS, PluginBridge } from "./PluginBridge";

function waitForOpen(socket: WebSocket): Promise<void> {
    return new Promise((resolve, reject) => {
        socket.once("open", () => resolve());
        socket.once("error", reject);
    });
}

function waitForClose(socket: WebSocket): Promise<{ code: number; reason: string }> {
    return new Promise((resolve) => {
        socket.once("close", (code, reason) => resolve({ code, reason: reason.toString() }));
    });
}

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function closeBridge(bridge: PluginBridge): Promise<void> {
    const internals = bridge as any;
    for (const connection of internals.connectedClients.values()) {
        connection.socket.close();
    }
    await new Promise<void>((resolve, reject) => {
        internals.wsServer.close((error: Error | undefined) => (error ? reject(error) : resolve()));
    });
}

test("replaces a frozen duplicate token connection with the new plugin connection", async () => {
    const mcpServer = {
        isMultiUserMode: () => true,
        getSessionContext: () => ({ userToken: "token-1" }),
    } as any;

    const bridge = new PluginBridge(mcpServer, 0);
    const port = ((bridge as any).wsServer.address() as { port: number }).port;

    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-1`);
    await waitForOpen(first);
    first.send(JSON.stringify({ type: "freeze" }));
    await delay(25);

    const second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-1`);
    await waitForOpen(second);
    await delay(25);

    try {
        assert.equal(second.readyState, WebSocket.OPEN);
        assert.notEqual(first.readyState, WebSocket.OPEN);
    } finally {
        first.close();
        second.close();
        await closeBridge(bridge);
    }
});

test("rejects a healthy duplicate token connection", async () => {
    const mcpServer = {
        isMultiUserMode: () => true,
        getSessionContext: () => ({ userToken: "token-2" }),
    } as any;

    const bridge = new PluginBridge(mcpServer, 0);
    const port = ((bridge as any).wsServer.address() as { port: number }).port;

    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-2`);
    await waitForOpen(first);

    const second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-2`);
    await waitForOpen(second);
    const close = await waitForClose(second);

    try {
        assert.equal(close.code, 1008);
        assert.match(close.reason, /Duplicate connection/);
        assert.equal(first.readyState, WebSocket.OPEN);
    } finally {
        first.close();
        second.close();
        await closeBridge(bridge);
    }
});

test("replaces a heartbeat-stale duplicate token connection with the new plugin connection", async () => {
    const mcpServer = {
        isMultiUserMode: () => true,
        getSessionContext: () => ({ userToken: "token-3" }),
    } as any;

    const bridge = new PluginBridge(mcpServer, 0);
    const port = ((bridge as any).wsServer.address() as { port: number }).port;

    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-3`);
    await waitForOpen(first);
    const connection = (bridge as any).clientsByToken.get("token-3");
    connection.lastHeartbeat = Date.now() - (HEARTBEAT_STALE_THRESHOLD_MS + 1_000);

    const second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-3`);
    await waitForOpen(second);
    await delay(25);

    try {
        assert.equal(second.readyState, WebSocket.OPEN);
        assert.notEqual(first.readyState, WebSocket.OPEN);
    } finally {
        first.close();
        second.close();
        await closeBridge(bridge);
    }
});
