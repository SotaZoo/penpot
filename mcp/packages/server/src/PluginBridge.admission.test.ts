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

async function waitForCondition(condition: () => boolean, timeoutMs: number = 500): Promise<void> {
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
        connection.socket.terminate();
    }
    await new Promise<void>((resolve, reject) => {
        internals.wsServer.close((error: Error | undefined) => (error ? reject(error) : resolve()));
    });
}

function createBridge(userToken: string, redisBridge?: any): PluginBridge {
    const mcpServer = {
        isMultiUserMode: () => true,
        getSessionContext: () => ({ userToken }),
    } as any;
    return new PluginBridge(mcpServer, 0, 30, redisBridge);
}

test("records a browser freeze and replaces that duplicate token owner", async () => {
    const bridge = createBridge("token-frozen");
    const port = ((bridge as any).wsServer.address() as { port: number }).port;
    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-frozen`);
    let second: WebSocket | undefined;

    try {
        await waitForOpen(first);
        first.send(JSON.stringify({ type: "freeze" }));
        await waitForCondition(() => (bridge as any).clientsByToken.get("token-frozen")?.frozen === true);

        second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-frozen`);
        await waitForOpen(second);
        await waitForCondition(() => first.readyState !== WebSocket.OPEN);

        assert.equal(second.readyState, WebSocket.OPEN);
        assert.notEqual(first.readyState, WebSocket.OPEN);
    } finally {
        first.close();
        second?.close();
        await closeBridge(bridge);
    }
});

test("rejects a healthy duplicate token connection without disturbing its owner", async () => {
    const subscriptions: string[] = [];
    const unsubscriptions: string[] = [];
    const redisBridge = {
        subscribeToTasks: async (userToken: string) => subscriptions.push(userToken),
        unsubscribeFromTasks: async (userToken: string) => unsubscriptions.push(userToken),
    } as any;
    const bridge = createBridge("token-healthy", redisBridge);
    const port = ((bridge as any).wsServer.address() as { port: number }).port;
    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-healthy`);
    let second: WebSocket | undefined;

    try {
        await waitForOpen(first);
        await waitForCondition(() => subscriptions.length === 1);
        const owner = (bridge as any).clientsByToken.get("token-healthy");
        second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-healthy`);
        const secondClose = waitForClose(second);
        await waitForOpen(second);
        const close = await secondClose;

        assert.equal(close.code, 1008);
        assert.match(close.reason, /Duplicate connection/);
        assert.equal(first.readyState, WebSocket.OPEN);
        assert.equal((bridge as any).clientsByToken.get("token-healthy"), owner);
        assert.deepEqual(subscriptions, ["token-healthy"]);
        assert.deepEqual(unsubscriptions, []);
    } finally {
        first.close();
        second?.close();
        await closeBridge(bridge);
    }
});

test("unsubscribes Redis routing when the current token owner closes", async () => {
    const subscriptions: string[] = [];
    const unsubscriptions: string[] = [];
    const redisBridge = {
        subscribeToTasks: async (userToken: string) => subscriptions.push(userToken),
        unsubscribeFromTasks: async (userToken: string) => unsubscriptions.push(userToken),
    } as any;
    const bridge = createBridge("token-cleanup", redisBridge);
    const port = ((bridge as any).wsServer.address() as { port: number }).port;
    const socket = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-cleanup`);

    try {
        await waitForOpen(socket);
        await waitForCondition(() => subscriptions.length === 1);
        socket.close();
        await waitForCondition(() => unsubscriptions.length === 1);

        assert.deepEqual(unsubscriptions, ["token-cleanup"]);
        assert.equal((bridge as any).clientsByToken.has("token-cleanup"), false);
    } finally {
        socket.close();
        await closeBridge(bridge);
    }
});

test("replaces a heartbeat-stale duplicate token owner", async () => {
    const bridge = createBridge("token-stale");
    const port = ((bridge as any).wsServer.address() as { port: number }).port;
    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-stale`);
    let second: WebSocket | undefined;

    try {
        await waitForOpen(first);
        const connection = (bridge as any).clientsByToken.get("token-stale");
        connection.lastHeartbeat = Date.now() - (HEARTBEAT_STALE_THRESHOLD_MS + 1_000);

        second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-stale`);
        await waitForOpen(second);
        await waitForCondition(() => first.readyState !== WebSocket.OPEN);

        assert.equal(second.readyState, WebSocket.OPEN);
        assert.notEqual((bridge as any).clientsByToken.get("token-stale"), connection);
    } finally {
        first.close();
        second?.close();
        await closeBridge(bridge);
    }
});

test("keeps Redis token routing intact while replacing a stale owner", async () => {
    const subscriptions: string[] = [];
    const unsubscriptions: string[] = [];
    const redisBridge = {
        subscribeToTasks: async (userToken: string) => subscriptions.push(userToken),
        unsubscribeFromTasks: async (userToken: string) => unsubscriptions.push(userToken),
    } as any;
    const bridge = createBridge("token-redis", redisBridge);
    const port = ((bridge as any).wsServer.address() as { port: number }).port;
    const first = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-redis`);
    let second: WebSocket | undefined;

    try {
        await waitForOpen(first);
        await waitForCondition(() => subscriptions.length === 1);
        (bridge as any).clientsByToken.get("token-redis").frozen = true;

        second = new WebSocket(`ws://127.0.0.1:${port}/mcp/ws?userToken=token-redis`);
        await waitForOpen(second);
        await waitForCondition(() => first.readyState !== WebSocket.OPEN);
        await waitForCondition(() => subscriptions.length === 2);

        assert.equal(second.readyState, WebSocket.OPEN);
        assert.deepEqual(unsubscriptions, []);
    } finally {
        first.close();
        second?.close();
        await closeBridge(bridge);
    }
});
