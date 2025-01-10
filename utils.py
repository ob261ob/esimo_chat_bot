class BaseLogger:
    def __init__(self) -> None:
        self.info = print


def extract_region_and_data(input_string):
    lines = input_string.strip().split("\n")

    region = ""
    data = ""
    is_data = False  # flag to know if we are inside a "Data" block

    for line in lines:
        if line.startswith("Region:"):
            region = line.split("Region: ", 1)[1].strip()
        elif line.startswith("Data:"):
            data = line.split("Data: ", 1)[1].strip()
            is_data = True  # set the flag to True once we encounter a "Data:" line
        elif is_data:
            # if the line does not start with "Data:" but we are inside a "Data" block,
            # then it is a continuation of the data
            data += "\n" + line.strip()

    return region, data


def create_vector_index(driver, dimension: int) -> None:
    index_query = "CALL db.index.vector.createNodeIndex('region_data', 'Data', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass


def create_constraints(driver):
    driver.query(
        "CREATE CONSTRAINT region_name IF NOT EXISTS FOR (r:Region) REQUIRE (r.name) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT data_id IF NOT EXISTS FOR (d:Data) REQUIRE (d.id) IS UNIQUE"
    )
