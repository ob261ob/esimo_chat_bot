#syntax = docker/dockerfile:1.4

FROM ollama/ollama:latest AS ollama
FROM babashka/babashka:latest

# Копируем ollama со всеми зависимостями
COPY --from=ollama /bin/ollama /bin/ollama
COPY --from=ollama /lib /lib
COPY --from=ollama /usr/lib /usr/lib
COPY --from=ollama /usr/bin /usr/bin

# Убедимся, что бинарник исполняемый
RUN chmod +x /bin/ollama

# Проверим зависимости
RUN ldd /bin/ollama || true

COPY <<EOF pull_model.clj
(ns pull-model
  (:require [babashka.process :as process]
            [clojure.core.async :as async]))

(try
  (let [llm (get (System/getenv) "LLM")
        url (get (System/getenv) "OLLAMA_BASE_URL")]
    (println (format "pulling ollama model %s using %s" llm url))
    (if (and llm 
         url 
         (not (#{"gpt-4" "gpt-3.5" "claudev2" "gpt-4o" "gpt-4-turbo"} llm))
         (not (some #(.startsWith llm %) ["ai21.jamba-instruct-v1:0"
                                          "amazon.titan"
                                          "anthropic.claude"
                                          "cohere.command"
                                          "meta.llama"
                                          "mistral.mi"])))

      (let [done (async/chan)
            proc (process/process {:env {"OLLAMA_HOST" url
                                        "HOME" "/tmp"}
                                  :out :inherit  ;; Это ключевое изменение
                                  :err :inherit}
                                 ["/bin/ollama" "pull" llm])]
        (async/go-loop [n 0]
          (let [[v _] (async/alts! [done (async/timeout 5000)])]
            (if (= :stop v) 
              :stopped 
              (do (println (format "... pulling model (%ss) - will take several minutes" (* n 10))) 
                  (recur (inc n))))))
        @proc  ;; Ждем завершения процесса
        (async/>!! done :stop))

      (println "OLLAMA model only pulled if both LLM and OLLAMA_BASE_URL are set and the LLM model is not gpt")))
  (catch Throwable e 
    (binding [*out* *err*]
      (println "Error:" e))
    (System/exit 1)))
EOF

ENTRYPOINT ["bb", "-f", "pull_model.clj"]