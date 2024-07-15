import networkx as nx
import html
import re
from typing import Any
import numbers
import matplotlib.pyplot as plt


import openai 
from openai import OpenAI

from CONSTANT import * 
from load_env import * 

class GraphRAG:


    def __init__(self) -> None:
        self.all_records: dict[int, str] = {}
        self.source_doc_map: dict[int, str] = {}

        self.client = OpenAI(api_key=OPENAI_KEY)


    def start(self):
        # NOTE: Gửi yêu cầu để sinh kết quả dạng graph 
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "assistant",
                    "content": GRAPH_EXTRACTION_PROMPT,
                }
            ],
            model="gpt-4o",
        ) 
        text = response.choices[0].message.content

        doc_index = 0
        source_doc_map[doc_index] = GRAPH_EXTRACTION_PROMPT
        all_records[doc_index] = text



    def search(self):
        pass 

    def clean_str(self, input: Any) -> str:
        """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
        # If we get non-string input, just give it back
        if not isinstance(input, str):
            return input

        result = html.unescape(input.strip())
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    def create_graph(self):
        record_delimiter = DEFAULT_RECORD_DELIMITER
        tuple_delimiter = DEFAULT_TUPLE_DELIMITER
        graph = nx.Graph()
        for source_doc_id, extracted_data in all_records.items():
            records = [r.strip() for r in extracted_data.split(record_delimiter)]

            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(tuple_delimiter)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = clean_str(record_attributes[1].upper())
                    entity_type = clean_str(record_attributes[2].upper())
                    entity_description = clean_str(record_attributes[3])

                    if entity_name in graph.nodes():
                        node = graph.nodes[entity_name]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["entity_type"] = (
                            entity_type if entity_type != "" else node["entity_type"]
                        )
                    else:
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )

                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    # add this record as edge
                    source = clean_str(record_attributes[1].upper())
                    target = clean_str(record_attributes[2].upper())
                    edge_description = clean_str(record_attributes[3])
                    edge_source_id = clean_str(str(source_doc_id))
                    weight = (
                        float(record_attributes[-1])
                        if isinstance(record_attributes[-1], numbers.Number)
                        else 1.0
                    )
                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )

    def view(self):
        
        pos = nx.spring_layout(graph)  # positions for all nodes
        # Draw the graph
        nx.draw(graph, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')


        # Display the graph
        plt.show()