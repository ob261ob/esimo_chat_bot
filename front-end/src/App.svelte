<script>
    import { tick } from "svelte";
    import SvelteMarkdown from "svelte-markdown";
    import botImage from "./assets/images/bot.jpeg";
    import meImage from "./assets/images/me.jpeg";
    import MdLink from "./lib/MdLink.svelte";
    import External from "./lib/External.svelte";
    import { chatStates, chatStore } from "./lib/chat.store.js";
    import Modal from "./lib/Modal.svelte";
    import { generationStore } from "./lib/generation.store";

    // Изменяем на строковый тип для трех вариантов
    let ragMode = "enabled"; // 'enabled', 'disabled', 'attribute'
    let question = "Отправь мне сводный бюллетень по районам ДВ морей";
    let shouldAutoScroll = true;
    let input;
    let senderImages = { bot: botImage, me: meImage };
    let generationModalOpen = false;
    
    function send() {
        // Преобразуем строковое значение в boolean для совместимости с chatStore
        // const ragBool = ragMode === "enabled" ? true : 
        //                ragMode === "disabled" ? false : null;
        chatStore.send(question, ragMode);
        question = "";
    }

    function handleAttributeMode() {
        // Логика для специального режима Attribute
        console.log("Attribute mode selected");
        // Здесь можно добавить специальную обработку
    }

    function scrollToBottom(node, _) {
        const scroll = () => node.scrollTo({ top: node.scrollHeight });
        scroll();
        return { update: () => shouldAutoScroll && scroll() };
    }

    function scrolling(e) {
        shouldAutoScroll = e.target.scrollTop + e.target.clientHeight > e.target.scrollHeight - 55;
    }

    function generateTicket(text) {
        generationStore.generate(text);
        generationModalOpen = true;
    }

    $: $chatStore.state === chatStates.IDLE && input && focus(input);
    async function focus(node) {
        await tick();
        node.focus();
    }
</script>

<main class="h-full text-sm bg-gradient-to-t from-indigo-100 bg-fixed overflow-hidden">
    <div on:scroll={scrolling} class="flex h-full flex-col py-12 overflow-y-auto" use:scrollToBottom={$chatStore}>
        <div class="w-4/5 mx-auto flex flex-col mb-32">
            {#each $chatStore.data as message (message.id)}
                <div
                    class="max-w-[80%] min-w-[40%] rounded-lg p-4 mb-4 overflow-x-auto bg-white border border-indigo-200"
                    class:self-end={message.from === "me"}
                    class:text-right={message.from === "me"}
                >
                    <div class="flex flex-row gap-2">
                        {#if message.from === "me"}
                            <button
                                aria-label="Generate a new internal ticket from this question"
                                title="Generate a new internal ticket from this question"
                                on:click={() => generateTicket(message.text)}
                                class="w-6 h-6 flex flex-col justify-center items-center border rounded border-indigo-200"
                                ><External --color="#ccc" --hover-color="#999" /></button
                            >
                        {/if}
                        <div
                            class:ml-auto={message.from === "me"}
                            class="relative w-12 h-12 border border-indigo-200 rounded flex justify-center items-center overflow-hidden"
                        >
                            <img src={senderImages[message.from]} alt="" class="rounded-sm" />
                        </div>
                        {#if message.from === "bot"}
                            <div class="text-sm">
                                <div>Model: {message.model ? message.model : ""}</div>
                                <div>RAG: {message.rag ? message.rag : ""}</div>
                            </div>
                        {/if}
                    </div>
                    <div class="mt-4"><SvelteMarkdown source={message.text} renderers={{ link: MdLink }} /></div>
                </div>
            {/each}
        </div>
        <div class="text-sm w-full fixed bottom-16">
            <div class="shadow-lg bg-indigo-50 rounded-lg w-4/5 xl:w-2/3 2xl:w-1/2 mx-auto">
                <div class="rounded-t-lg px-4 py-2 font-light">
                    <div class="font-semibold">RAG mode</div>
                    <div class="flex items-center gap-4">
                        <label class="flex items-center gap-1">
                            <input type="radio" bind:group={ragMode} value="disabled" /> Disabled
                        </label>
                        <label class="flex items-center gap-1">
                            <input type="radio" bind:group={ragMode} value="enabled" /> Enabled
                        </label>
                        <label class="flex items-center gap-1">
                            <input type="radio" bind:group={ragMode} value="attribute" /> Attribute
                        </label>
                    </div>
                </div>
                <form class="rounded-md w-full bg-white p-2 m-0" on:submit|preventDefault={send}>
                    <input
                        placeholder="Напишите какие данные и в каком регионе вас интересуют"
                        disabled={$chatStore.state === chatStates.RECEIVING}
                        class="text-lg w-full bg-white focus:outline-none px-4"
                        bind:value={question}
                        bind:this={input}
                        type="text"
                    />
                </form>
            </div>
        </div>
    </div>
</main>

{#if generationModalOpen}
    <Modal title="my title" text="my text" on:close={() => (generationModalOpen = false)} />
{/if}

<style>
    :global(pre) {
        @apply bg-gray-100 rounded-lg p-4 border border-indigo-200;
    }
    :global(code) {
        @apply text-indigo-500;
    }
</style>