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

async function waitForCondition(condition: () => boolean, timeoutMs: number = 1_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (!condition()) {
        if (Date.now() >= deadline) {
            throw new Error("Timed out waiting for condition");
        }
        await delay(5);
    }
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
    const connection = (bridge as any).clientsByToken.get("token-1");
    await waitForCondition(() => connection.frozen);

    const firstClose = waitForClose(first);
    const second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-1`);
    await waitForOpen(second);
    await firstClose;

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
    const secondClose = waitForClose(second);
    await waitForOpen(second);
    const close = await secondClose;

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

    const firstClose = waitForClose(first);
    const second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-3`);
    await waitForOpen(second);
    await firstClose;

    try {
        assert.equal(second.readyState, WebSocket.OPEN);
        assert.notEqual(first.readyState, WebSocket.OPEN);
    } finally {
        first.close();
        second.close();
        await closeBridge(bridge);
    }
});

test("preserves the Redis token subscription while replacing a stale connection", async () => {
    const subscriptions: string[] = [];
    const unsubscriptions: string[] = [];
    const redisBridge = {
        subscribeToTasks: async (userToken: string) => {
            subscriptions.push(userToken);
        },
        unsubscribeFromTasks: async (userToken: string) => {
            unsubscriptions.push(userToken);
        },
    } as any;
    const mcpServer = {
        isMultiUserMode: () => true,
        getSessionContext: () => ({ userToken: "token-4" }),
    } as any;

    const bridge = new PluginBridge(mcpServer, 0, redisBridge);
    const port = ((bridge as any).wsServer.address() as { port: number }).port;

    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-4`);
    await waitForOpen(first);
    await waitForCondition(() => subscriptions.length === 1);
    const connection = (bridge as any).clientsByToken.get("token-4");
    connection.frozen = true;

    const firstClose = waitForClose(first);
    const second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-4`);
    await waitForOpen(second);
    await firstClose;
    await waitForCondition(() => subscriptions.length === 2);

    try {
        assert.equal(second.readyState, WebSocket.OPEN);
        assert.deepEqual(unsubscriptions, []);
    } finally {
        first.close();
        second.close();
        await closeBridge(bridge);
    }
});
