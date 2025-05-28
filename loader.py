import os
import requests
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
import streamlit as st
from streamlit.logger import get_logger
from datetime import datetime
import json
import streamlit as st
load_dotenv(".env")
from chains import (
    load_embedding_model,
    load_llm
)
from utils import (
    create_constraints,
    create_vector_index,
    BaseLogger,
)
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM") or "llama3"
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

create_constraints(neo4j_graph)
create_vector_index(neo4j_graph, dimension)

llm = load_llm(
    llm_name, logger=BaseLogger(), config={"ollama_base_url": ollama_base_url}
)
def generate_region(description: str) -> str:
    """Генерирует регион на основе описания события с помощью локальной Ollama"""
    
    prompt = f"""
    Определи, в каком регионе России происходит это событие, на основе описания.
    Ответь только названием региона (например, "Москва", "Санкт-Петербург", 
    "Краснодарский край"). Можешь так же писать нзавание моря, федельного округа или океана. Если не можешь определить, верни "Неизвестно".
    
    Описание события: {description}
    Регион:
    """
    
    try:
        region = llm.invoke(prompt)
        return region.content.strip()  # Убираем лишние кавычки и пробелы
    except Exception as e:
        st.error(f"Ошибка при генерации региона: {e}")
        return "Неизвестно"
    
# def load_so_data(tag: str = "neo4j", page: int = 1) -> None:
#     parameters = (
#         f"?pagesize=100&page={page}&order=desc&sort=creation&answers=1&tagged={tag}"
#         "&site=stackoverflow&filter=!*236eb_eL9rai)MOSNZ-6D3Q6ZKb0buI*IVotWaTb"
#     )
#     data = requests.get(so_api_base_url + parameters).json()
#     insert_so_data(data)


# def load_high_score_so_data() -> None:
#     parameters = (
#         f"?fromdate=1664150400&order=desc&sort=votes&site=stackoverflow&"
#         "filter=!.DK56VBPooplF.)bWW5iOX32Fh1lcCkw1b_Y6Zkb7YD8.ZMhrR5.FRRsR6Z1uK8*Z5wPaONvyII"
#     )
#     data = requests.get(so_api_base_url + parameters).json()
#     insert_so_data(data)


# def insert_so_data(data: dict) -> None:
#     # Calculate embedding values for questions and answers
#     for q in data["items"]:
#         question_text = q["title"] + "\n" + q["body_markdown"]
#         q["embedding"] = embeddings.embed_query(question_text)
#         for a in q["answers"]:
#             a["embedding"] = embeddings.embed_query(
#                 question_text + "\n" + a["body_markdown"]
#             )

#     # Cypher, the query language of Neo4j, is used to import the data
#     # https://neo4j.com/docs/getting-started/cypher-intro/
#     # https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise/
#     import_query = """
#     UNWIND $data AS q
#     MERGE (question:Question {id:q.question_id}) 
#     ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
#         question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
#         question.body = q.body_markdown, question.embedding = q.embedding
#     FOREACH (tagName IN q.tags | 
#         MERGE (tag:Tag {name:tagName}) 
#         MERGE (question)-[:TAGGED]->(tag)
#     )
#     FOREACH (a IN q.answers |
#         MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
#         SET answer.is_accepted = a.is_accepted,
#             answer.score = a.score,
#             answer.creation_date = datetime({epochSeconds:a.creation_date}),
#             answer.body = a.body_markdown,
#             answer.embedding = a.embedding
#         MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
#         ON CREATE SET answerer.display_name = a.owner.display_name,
#                       answerer.reputation= a.owner.reputation
#         MERGE (answer)<-[:PROVIDED]-(answerer)
#     )
#     WITH * WHERE NOT q.owner.user_id IS NULL
#     MERGE (owner:User {id:q.owner.user_id})
#     ON CREATE SET owner.display_name = q.owner.display_name,
#                   owner.reputation = q.owner.reputation
#     MERGE (owner)-[:ASKED]->(question)
#     """
#     neo4j_graph.query(import_query, {"data": data["items"]})

def insert_so_data(data: list) -> None:
    if not isinstance(data, list):
        raise ValueError("Ожидается список данных.")

    if len(data) == 0:
        st.warning("Список данных пуст.")
        return

    progress_bar = st.progress(0, text="Подготовка данных...")

    processed_data = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("Каждый элемент данных должен быть словарем.")
        
        required_keys = ["id", "title", "link", "start_date", "end_date", "description", "region", "embedding"]
        for key in required_keys:
            if key not in item:
                raise ValueError(f"Отсутствует ключ: {key} в элементе данных {item}")

        # Обновляем embedding заново
        item["embedding"] = embeddings.embed_query(item["title"])

        # Обработка региона
        raw_region = item.get("region", "")
        clean_region = raw_region.strip("[]")
        region_list = [r.strip() for r in clean_region.split(",") if r.strip()]
        item["region_list"] = region_list

        processed_data.append(item)

        # 🔽 Показываем, что импортируется
        st.markdown(f"➡️ Импортируется: **{item['title']}**")

        progress = int((idx + 1) / len(data) * 50)
        progress_bar.progress(progress, text=f"Подготовка: {progress}%")

    # Цикл импорта в Neo4j
    import_query = """
    UNWIND $data AS item
    MERGE (data:Data {id: item.id})
    ON CREATE SET 
        data.title = item.title, 
        data.link = item.link, 
        data.start_date = datetime({epochSeconds: item.start_date}), 
        data.end_date = datetime({epochSeconds: item.end_date}), 
        data.description = item.description, 
        data.embedding = item.embedding
    WITH data, item.region_list AS regions
    UNWIND regions AS region_name
    MERGE (region:Region {name: region_name})
    MERGE (data)-[:LOCATED_IN]->(region)
    """

    batch_size = 10
    for i in range(0, len(processed_data), batch_size):
        batch = processed_data[i:i+batch_size]
        neo4j_graph.query(import_query, {"data": batch})
        progress = 50 + int((i + batch_size) / len(processed_data) * 50)
        progress_bar.progress(min(progress, 100), text=f"Импорт: {min(progress, 100)}%")

    progress_bar.progress(100, text="Импорт завершён ✅")





# Streamlit
def get_tag() -> str:
    input_text = st.text_input(
        "Which tag questions do you want to import?", value="neo4j"
    )
    return input_text


def get_pages():
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input(
            "Number of pages (100 questions per page)", step=1, min_value=1
        )
    with col2:
        start_page = st.number_input("Start page", step=1, min_value=1)
    st.caption("Only questions with answers will be imported.")
    return (int(num_pages), int(start_page))


# def render_page():
#     datamodel_image = Image.open("./images/datamodel.png")
#     st.header("StackOverflow Loader")
#     st.subheader("Choose StackOverflow tags to load into Neo4j")
#     st.caption("Go to http://localhost:7474/ to explore the graph.")

#     user_input = get_tag()
#     num_pages, start_page = get_pages()

#     if st.button("Import", type="primary"):
#         with st.spinner("Loading... This might take a minute or two."):
#             try:
#                 for page in range(1, num_pages + 1):
#                     load_so_data(user_input, start_page + (page - 1))
#                 st.success("Import successful", icon="✅")
#                 st.caption("Data model")
#                 st.image(datamodel_image)
#                 st.caption("Go to http://localhost:7474/ to interact with the database")
#             except Exception as e:
#                 st.error(f"Error: {e}", icon="🚨")
#     with st.expander("Highly ranked questions rather than tags?"):
#         if st.button("Import highly ranked questions"):
#             with st.spinner("Loading... This might take a minute or two."):
#                 try:
#                     load_high_score_so_data()
#                     st.success("Import successful", icon="✅")
#                 except Exception as e:
#                     st.error(f"Error: {e}", icon="🚨")



def render_page():
    st.header("ESIMO Data Loader — Ввод событий вручную")

    if "events" not in st.session_state:
        st.session_state["events"] = []

    st.subheader("Добавьте новое событие")
    with st.form(key=f"form_add_event_{len(st.session_state['events'])}"):

        event_id = st.text_input("ID События", key="event_id")
        title = st.text_input("Название", key="title")
        link = st.text_input("Ссылка", key="link")
        start_date = st.text_input("Дата начала (в формате YYYY-MM-DD)", key="start_date")
        end_date = st.text_input("Дата окончания (в формате YYYY-MM-DD)", key="end_date")
        description = st.text_area("Описание", key="description")
        region = st.text_input("Регион", key="region")

        submit_event = st.form_submit_button(label="Добавить событие")
        embedding = embeddings.embed_query(title)
        if submit_event:
            st.session_state["events"].append({
                "id": event_id,
                "title": title,
                "link": link,
                "start_date": start_date,
                "end_date": end_date,
                "description": description,
                "region": region,
                "embedding": embedding
            })
            st.success("Событие добавлено в список для импорта.", icon="✅")

    st.subheader("Список событий для импорта")
    if st.session_state["events"]:
        for idx, event in enumerate(st.session_state["events"]):
            with st.expander(f"Событие {idx + 1}: {event['title']}"):
                st.write("ID:", event["id"])
                st.write("Название:", event["title"])
                st.write("Ссылка:", event["link"])
                st.write("Дата начала:", event["start_date"])
                st.write("Дата окончания:", event["end_date"])
                st.write("Описание:", event["description"])
                st.write("Регион:", event["region"])

    # Кнопка для загрузки данных из файла JSON
    st.subheader("Выгрузить события из файла JSON")
    uploaded_file = st.file_uploader("Загрузите файл JSON", type=["json"])

    if uploaded_file:
        try:
            file_data = json.load(uploaded_file)
            progress_bar = st.progress(0)
            total = len(file_data)

            for idx, event_data in enumerate(file_data):
                # embedding = embeddings.embed_query(event_data["title"])  # Заполняем embedding
                embedding = ""
                st.session_state["events"].append({
                    "id": event_data.get("identifier", ""),
                    "title": event_data["title"],
                    "link": event_data["link"],
                    "start_date": int(datetime.strptime(event_data["start_date"].split("T")[0], "%Y-%m-%d").timestamp()),
                    "end_date": int(datetime.strptime(event_data["end_date"].split("T")[0], "%Y-%m-%d").timestamp()),
                    "description": event_data["description"],
                    "region": event_data.get("geographic_coverage", {}).get("geo_object_name", "Неизвестно"),
                    "embedding": embedding
                })
                progress_bar.progress((idx + 1) / total)

            progress_bar.empty()
            st.success("События успешно добавлены из файла!", icon="✅")
        except json.JSONDecodeError:
            st.error("Ошибка при чтении файла. Убедитесь, что файл в формате JSON.")
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")


    # Кнопка для импорта всех событий
    if st.button("Импортировать все события"):
        if st.session_state["events"]:
            data = st.session_state["events"]
            with st.spinner("Импорт данных..."):
                try:
                    insert_so_data(data)
                    st.success("Все данные успешно импортированы!", icon="✅")
                    st.session_state["events"] = []  # Очищаем после импорта
                except Exception as e:
                    st.error(f"Переделывай: {e}")
        else:
            st.error("Нет событий для импорта.")

# Вызов функции
render_page()
