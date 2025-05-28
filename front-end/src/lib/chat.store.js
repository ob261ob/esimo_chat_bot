import { writable } from "svelte/store";

const API_ENDPOINT = "https://api.esimochatbot.ru/query-stream";

export const chatStates = {
    IDLE: "idle",
    RECEIVING: "receiving",
};

function createChatStore() {
    const { subscribe, update } = writable({ state: chatStates.IDLE, data: [] });
    
    function addMessage(from, text, rag) {
        const newId = Math.random().toString(36).substring(2, 9);
        update((state) => {
            const message = { id: newId, from, text, rag };
            state.data.push(message);
            return state;
        });
        return newId;
    }

    function updateMessage(existingId, text, model = null) {
        if (!existingId) {
            return;
        }
        update((state) => {
            const messages = state.data;
            const existingIdIndex = messages.findIndex((m) => m.id === existingId);
            if (existingIdIndex === -1) {
                return state;
            }
            messages[existingIdIndex].text += text;
            if (model) {
                messages[existingIdIndex].model = model;
            }
            return { ...state, data: messages };
        });
    }

    async function send(question, ragMode) {
        if (!question.trim().length) {
            return;
        }
        update((state) => ({ ...state, state: chatStates.RECEIVING }));
        addMessage("me", question, ragMode);
        const messageId = addMessage("bot", "", ragMode);
        try {
            const evt = new EventSource(`${API_ENDPOINT}?text=${encodeURI(question)}&rag=${ragMode}`);
            question = "";
            evt.onmessage = (e) => {
                if (e.data) {
                    const data = JSON.parse(e.data);
                    if (data.init) {
                        updateMessage(messageId, "", data.model);
                        return;
                    }
                    updateMessage(messageId, data.token);
                }
            };
            evt.onerror = (e) => {
                // Stream will end with an error
                // and we want to close the connection on end (otherwise it will keep reconnecting)
                evt.close();
                update((state) => ({ ...state, state: chatStates.IDLE }));
            };
        } catch (e) {
            updateMessage(messageId, "Error: " + e.message);
            update((state) => ({ ...state, state: chatStates.IDLE }));
        }
    }
    addMessage("bot", "Привет! Я чат-бот, который поможет тебе с поиском ресурсов веб-портала ЕСИМО. Я работаю в трёх режимах: 'disable' - отвечать будет только языковая модель Ollama, 'enable' - отвечать будет та же языковая модель, только с векторным поиском среди ресурсов ЕСИМО, 'attribute' - в разработке", null);
    return {
        subscribe,
        send,
    };
}

export const chatStore = createChatStore();
