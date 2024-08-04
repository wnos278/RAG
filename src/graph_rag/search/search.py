import pandas as pd
from llama_index.core import Document
import re  # Import thư viện re
from llama_index.core.graph_stores import SimplePropertyGraphStore  # Import SimplePropertyGraphStore
import networkx as nx  # Import networkx với tên gọi nx
from graspologic.partition import hierarchical_leiden  # Import hierarchical_leiden từ graspologic.partition

from llama_index.core.llms import ChatMessage  # Import ChatMessage



sample_document = """
Hưng Đạo Vương Trần Quốc Tuấn là người đã trực tiếp chỉ huy quân đội đánh tan hai cuộc xâm lược của quân Nguyên–Mông năm 1285 và năm 1288. Ông được biết đến với những chiến công lẫy lừng, là một nhà quân sự tài ba với gia thế hiển hách. Đúng là người giỏi, người tài nên cái gì ông cũng dám làm, dám chịu. Đau khổ vì người con gái mình yêu thương phải lấy người khác làm chồng, ông đã tạo ra một phi vụ cướp dâu chấn động nhất lịch sử Việt Nam.


Năm 1237, Trần Cảnh (Trần Thái Tông) lên ngôi đã lâu mà vẫn không có con nối dõi, lo sợ nhà Trần bị tuyệt hậu, Trần Thủ Độ đã gây sức ép phế Lý Chiêu Hoàng để ép Trần Cảnh kết hôn với chị dâu của mình là công chúa Thuận Thiên, tức là vợ của Trần Liễu - cha của Trần Quốc Tuấn. Trần Liễu tức giận, mang binh rửa hận nhưng thân già sức yếu nên việc bất thành, cuối cùng phải buông giáp quy hàng, bị giáng xuống làm An Sinh Vương, cho về an trú ở đất Yên Sinh. Khi ấy, Trần Quốc Tuấn mới 7 tuổi.

Chị của Trần Cảnh là Thụy Bà công chúa vì thương cháu mình đang còn nhỏ phải rời kinh đô tới nơi xa, đã cầu xin vua để nhận nuôi Quốc Tuấn để khuây khỏa nỗi buồn khi chồng bà đã mất.

Bà nhận nuôi Quốc Tuấn được 8 năm, cho ông học văn, học võ, lớn lên với các con em hoàng tộc cùng trang lứa. Cũng chính trong thời gian này, Trần Quốc Tuấn gặp gỡ, cùng trải qua thời niên thiếu của mình với Thiên Thành công chúa.

Trong suốt những năm tháng học tập và sinh sống nơi cung cấm, tình cảm của công chúa và Trần Quốc Tuấn cứ lớn dần lên, quấn quýt không rời. Mối tình thanh mai trúc mã cứ thế nở rộ.

Cứ tưởng đây là mối lương duyên trời ban, cho tới khi Thiên Thành tới tuổi gả chồng, vua Trần Thái Tông đã hạ chỉ gả nàng cho Trung Thành Vương, con trai của Nhân Đạo Vương, phá tan giấc mộng đôi lứa của hai người.

Sách Đại Việt sử ký toàn thư ghi lại rằng ngày 15 tháng 2 năm 1251, vua mở hội lớn 7 ngày đêm, bày các tranh về lễ kết tóc và nhiều trò chơi cho người trong triều ngoài nội đến xem, ý muốn cho công chúa Thiên Thành làm lễ kết tóc với Trung Thành Vương. Trước đó, nhà vua cũng cho Thiên Thành công chúa về ở vương phủ cha của Trung Thành Vương để chờ ngày làm lễ ăn hỏi.

Trong khi cả kinh thành đang tưng bừng với những trò chơi và lễ hội, ở vương phủ Trần Quốc Tuấn chỉ cần nghĩ đến việc ngày mai, người con gái mình yêu thương sẽ trở thành vợ người khác thì tâm tư của chàng càng đau đớn. Chàng trằn trọc suốt đêm không ngủ cuối cùng chàng đưa ra quyết định táo bạo đó chính là đột nhập vào phủ Nhân Đạo Vương, cướp vợ về.

Nghĩ là làm, trong đêm tối, nhân lúc mọi người còn đang say mê với lễ hội, Trần Quốc Tuấn lẻn vào phủ Nhân Đạo Vương. Biết không thể theo vào bằng cửa chính, chàng đã tìm cách trèo tường, vượt qua hàng toán lính tuần tra, dò trong đêm đen và tìm được chính xác phòng công chúa.

Trái tim đau khổ của Thiên Thành sống lại lần nữa khi thấy người tình trong mộng xuất hiện trước mặt mình. Khi ấy, cả phủ Nhân Đạo Vương vẫn đang say trong lễ hội, không ai biết, trong phòng công chúa, đôi uyên ương đã gặp lại nhau. Thế nhưng, sự liều lĩnh này của Trần Quốc Tuấn sẽ trở thành thảm án nếu sự vụ bị bại lộ. Và nếu như chuyện không bại lộ, thì hôm sau công chúa Thiên Thành phải kết hôn với con trai của Nhân Đạo Vương. Để tránh khỏi tai ương đó, Trần Quốc Tuấn đã đi tiếp một bước cờ cao minh, đó chính là dồn nhà vua vào thế sự đã rồi.

Ngay sau khi đột nhập thành công vào phòng công chúa, việc đầu tiên Trần Quốc Tuấn làm là ra lệnh cho thị nữ của công chúa về báo cho Thụy Bà công chúa, mẹ nuôi của chàng, Sau khi nhận được tin báo, Thụy Bà công chúa vào cung ngay lập tức và than khóc với Thái Tông: "Không ngờ Quốc Tuấn càn rỡ đang đêm lẻn vào chỗ của Thiên Thành. Nhân Đạo Vương đã bắt giữ hắn rồi, e sẽ giết hắn mất. Xin bệ hạ rủ lòng thương, sai người đến cứu".

Lời nói của Thụy Bà công chúa như sét đánh ngang tai nhà vua, Trần Thái Tông tức Cảnh lúc bấy giờ đã nhận đủ lễ vật của Nhân Đạo Vương, sao có thể để Trần Quốc Tuấn cả gan làm loạn như vậy? Thụy Bà công chúa tiếp tục kiên trì van xin. Cộng thêm với việc ông nghĩ rằng đó là huyết mạch của anh trai Trần Liễu, Thái Tông đã đã sai người vây phủ Nhân Đạo Vương, xông thẳng tới hoa viên vắng lặng, vào phòng công chúa Thiên Thành để áp giải, thực chất là hộ tống, Trần Quốc Tuấn ra ngoài một cách an toàn. Đến lúc đó, cả phủ Nhân Đạo Vương mới ngỡ ngàng nhận ra Trần Quốc Tuấn đã vào phủ "tư thông" với công chúa Thiên Thành.

Việc công chúa "tư thông" với nam tử khác ngay trong phủ sắp cưới là điều không thể chấp nhận được. Hôm sau, Thụy Bà công chúa đã nhanh tay hỏi cưới công chúa Thiên Thành cho cháu trai mình, với sinh lễ là 10 mâm vàng sống và nói "vì vội quá nên không sắm đủ lễ vật, mong hoàng thượng nhận cho". Trước chuyện đã rồi, Trần Thái Tông đành xuống chiếu gả Thiên Thành công chúa cho Trần Quốc Tuấn và ngậm ngùi cắt 2.000 khoảnh ruộng tốt ở huyện Ứng Thiên để "an ủi" Nhân Đạo Vương.

Cuối cùng Trần Quốc Tuấn chàng đã lấy được thanh mai trúc mã mà ông yêu bấy lâu. Hai vợ chồng chàng đã có một cuộc sống êm ấm, hạnh phúc, sinh được bốn trai, một gái. Bốn người con trai ai cũng không phụ danh tiếng người cha, đều là những danh tướng lẫy lừng nhà Trần. Người con gái út sau này trở thành Bảo Thánh Hoàng Hậu Trần Trinh, vợ vua Trần Nhân Tông, mẹ đẻ vua Trần Anh Tông.


"""
documents = [
    Document(text=sample_document)
]


import os
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("open_ai_key")

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4")

import asyncio  # Import asyncio
import nest_asyncio  # Import nest_asyncio

nest_asyncio.apply()  # Apply nest_asyncio

from typing import Any, List, Callable, Optional, Union, Dict  # Import các kiểu dữ liệu cần thiết
from IPython.display import Markdown, display  # Import các hàm để hiển thị

from llama_index.core.async_utils import run_jobs  # Import run_jobs từ llama_index
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,  # Import default_parse_triplets_fn
)
from llama_index.core.graph_stores.types import (
    EntityNode,  # Import EntityNode
    KG_NODES_KEY,  # Import KG_NODES_KEY
    KG_RELATIONS_KEY,  # Import KG_RELATIONS_KEY
    Relation,  # Import Relation
)
from llama_index.core.llms.llm import LLM  # Import LLM
from llama_index.core.prompts import PromptTemplate  # Import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,  # Import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
)
from llama_index.core.schema import TransformComponent, BaseNode  # Import TransformComponent, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field  # Import BaseModel, Field


class GraphRAGExtractor(TransformComponent):  # Định nghĩa lớp GraphRAGExtractor
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM  # Định nghĩa biến llm kiểu LLM
    extract_prompt: PromptTemplate  # Định nghĩa biến extract_prompt kiểu PromptTemplate
    parse_fn: Callable  # Định nghĩa biến parse_fn kiểu Callable
    num_workers: int  # Định nghĩa biến num_workers kiểu int
    max_paths_per_chunk: int  # Định nghĩa biến max_paths_per_chunk kiểu int

    def __init__(  # Định nghĩa hàm khởi tạo
        self,
        llm: Optional[LLM] = None,  # Định nghĩa tham số llm
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,  # Định nghĩa tham số extract_prompt
        parse_fn: Callable = default_parse_triplets_fn,  # Định nghĩa tham số parse_fn
        max_paths_per_chunk: int = 10,  # Định nghĩa tham số max_paths_per_chunk
        num_workers: int = 4,  # Định nghĩa tham số num_workers
    ) -> None:  # Trả về None
        """Init params."""  # Khởi tạo các tham số.
        from llama_index.core import Settings  # Import Settings từ llama_index

        if isinstance(extract_prompt, str):  # Kiểm tra nếu extract_prompt là str
            extract_prompt = PromptTemplate(extract_prompt)  # Chuyển đổi extract_prompt thành PromptTemplate

        super().__init__(  # Gọi hàm khởi tạo của lớp cha
            llm=llm or Settings.llm,  # Gán giá trị cho llm
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,  # Gán giá trị cho extract_prompt
            parse_fn=parse_fn,  # Gán giá trị cho parse_fn
            num_workers=num_workers,  # Gán giá trị cho num_workers
            max_paths_per_chunk=max_paths_per_chunk,  # Gán giá trị cho max_paths_per_chunk
        )

    @classmethod  # Định nghĩa phương thức class_name
    def class_name(cls) -> str:  # Trả về tên lớp
        return "GraphExtractor"

    def __call__(  # Định nghĩa phương thức __call__
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:  # Trả về danh sách BaseNode
        """Extract triples from nodes."""  # Trích xuất triples từ nodes.
        return asyncio.run(  # Chạy phương thức asyncio.run
            self.acall(nodes, show_progress=show_progress, **kwargs)  # Gọi phương thức acall
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:  # Định nghĩa phương thức bất đồng bộ _aextract
        """Extract triples from a node."""  # Trích xuất triples từ một node.
        assert hasattr(node, "text")  # Kiểm tra nếu node có thuộc tính text

        text = node.get_content(metadata_mode="llm")  # Lấy nội dung của node
        try:
            llm_response = await self.llm.apredict(  # Dự đoán kết quả với llm
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)  # Phân tích kết quả
        except ValueError:  # Bắt lỗi ValueError
            entities = []  # Gán giá trị mặc định cho entities
            entities_relationship = []  # Gán giá trị mặc định cho entities_relationship

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])  # Lấy giá trị của existing_nodes
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])  # Lấy giá trị của existing_relations
        metadata = node.metadata.copy()  # Sao chép metadata
        for entity, entity_type, description in entities:  # Lặp qua các entity
            metadata[
                "entity_description"
            ] = description  # Không sử dụng trong bản hiện tại. Sẽ hữu ích trong tương lai.
            entity_node = EntityNode(  # Tạo EntityNode
                name=entity, label=entity_type, properties=metadata
            )
            existing_nodes.append(entity_node)  # Thêm entity_node vào existing_nodes

        metadata = node.metadata.copy()  # Sao chép lại metadata
        for triple in entities_relationship:  # Lặp qua các triple
            subj, rel, obj, description = triple  # Lấy các giá trị từ triple
            subj_node = EntityNode(name=subj, properties=metadata)  # Tạo subj_node
            obj_node = EntityNode(name=obj, properties=metadata)  # Tạo obj_node
            metadata["relationship_description"] = description  # Gán giá trị cho relationship_description
            rel_node = Relation(  # Tạo Relation
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])  # Thêm subj_node và obj_node vào existing_nodes
            existing_relations.append(rel_node)  # Thêm rel_node vào existing_relations

        node.metadata[KG_NODES_KEY] = existing_nodes  # Gán lại giá trị cho KG_NODES_KEY
        node.metadata[KG_RELATIONS_KEY] = existing_relations  # Gán lại giá trị cho KG_RELATIONS_KEY
        return node  # Trả về node

    async def acall(  # Định nghĩa phương thức bất đồng bộ acall
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:  # Trả về danh sách BaseNode
        """Extract triples from nodes async."""  # Trích xuất triples từ nodes bất đồng bộ.
        jobs = []  # Khởi tạo danh sách jobs
        for node in nodes:  # Lặp qua các node
            jobs.append(self._aextract(node))  # Thêm job vào danh sách

        return await run_jobs(  # Chạy các job
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )






class GraphRAGStore(SimplePropertyGraphStore):  # Định nghĩa lớp GraphRAGStore kế thừa từ SimplePropertyGraphStore
    community_summary = {}  # Khởi tạo biến community_summary
    max_cluster_size = 5  # Đặt giá trị max_cluster_size là 5

    def generate_community_summary(self, text):  # Định nghĩa hàm generate_community_summary
        """Generate summary for a given text using an LLM."""  # Tạo bản tóm tắt cho văn bản đã cho bằng LLM.
        messages = [  # Khởi tạo danh sách messages
            ChatMessage(  # Tạo tin nhắn hệ thống
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),  # Tạo tin nhắn của người dùng
        ]
        response = OpenAI().chat(messages)  # Gửi tin nhắn tới OpenAI và nhận phản hồi
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()  # Làm sạch phản hồi nhận được
        return clean_response  # Trả về phản hồi đã làm sạch

    def build_communities(self):  # Định nghĩa hàm build_communities
        """Builds communities from the graph and summarizes them."""  # Xây dựng các cộng đồng từ đồ thị và tóm tắt chúng.
        nx_graph = self._create_nx_graph()  # Tạo đồ thị NetworkX
        community_hierarchical_clusters = hierarchical_leiden(  # Áp dụng thuật toán hierarchical_leiden
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(  # Thu thập thông tin chi tiết về cộng đồng
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)  # Tóm tắt các cộng đồng

    def _create_nx_graph(self):  # Định nghĩa hàm _create_nx_graph
        """Converts internal graph representation to NetworkX graph."""  # Chuyển đổi biểu diễn đồ thị nội bộ thành đồ thị NetworkX.
        nx_graph = nx.Graph()  # Khởi tạo đồ thị NetworkX
        for node in self.graph.nodes.values():  # Lặp qua các nút trong đồ thị
            nx_graph.add_node(str(node))  # Thêm nút vào đồ thị NetworkX
        for relation in self.graph.relations.values():  # Lặp qua các quan hệ trong đồ thị
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )  # Thêm cạnh vào đồ thị NetworkX
        return nx_graph  # Trả về đồ thị NetworkX

    def _collect_community_info(self, nx_graph, clusters):  # Định nghĩa hàm _collect_community_info
        """Collect detailed information for each node based on their community."""  # Thu thập thông tin chi tiết cho từng nút dựa trên cộng đồng của chúng.
        community_mapping = {item.node: item.cluster for item in clusters}  # Tạo ánh xạ giữa nút và cộng đồng
        community_info = {}  # Khởi tạo biến community_info
        for item in clusters:  # Lặp qua các phần tử trong clusters
            cluster_id = item.cluster  # Lấy ID của cộng đồng
            node = item.node  # Lấy nút
            if cluster_id not in community_info:  # Kiểm tra nếu cộng đồng chưa có trong community_info
                community_info[cluster_id] = []  # Tạo danh sách cho cộng đồng mới

            for neighbor in nx_graph.neighbors(node):  # Lặp qua các nút lân cận
                if community_mapping[neighbor] == cluster_id:  # Kiểm tra nếu lân cận thuộc cùng cộng đồng
                    edge_data = nx_graph.get_edge_data(node, neighbor)  # Lấy dữ liệu cạnh
                    if edge_data:  # Kiểm tra nếu có dữ liệu cạnh
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"  # Tạo chuỗi chi tiết
                        community_info[cluster_id].append(detail)  # Thêm chi tiết vào community_info
        return community_info  # Trả về community_info

    def _summarize_communities(self, community_info):  # Định nghĩa hàm _summarize_communities
        """Generate and store summaries for each community."""  # Tạo và lưu trữ các bản tóm tắt cho từng cộng đồng.
        for community_id, details in community_info.items():  # Lặp qua các cộng đồng trong community_info
            details_text = (
                "\n".join(details) + "."  # Đảm bảo kết thúc bằng dấu chấm
            )
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)  # Tạo bản tóm tắt và lưu trữ

    def get_community_summaries(self):  # Định nghĩa hàm get_community_summaries
        """Returns the community summaries, building them if not already done."""  # Trả về các bản tóm tắt cộng đồng, xây dựng chúng nếu chưa có.
        if not self.community_summary:  # Kiểm tra nếu chưa có bản tóm tắt
            self.build_communities()  # Xây dựng các cộng đồng
        return self.community_summary  # Trả về bản tóm tắt cộng đồng


from llama_index.core.query_engine import CustomQueryEngine  # Import CustomQueryEngine từ llama_index
from llama_index.core.llms import LLM  # Import LLM từ llama_index


class GraphRAGQueryEngine(CustomQueryEngine):  # Định nghĩa lớp GraphRAGQueryEngine kế thừa từ CustomQueryEngine
    graph_store: GraphRAGStore  # Định nghĩa biến graph_store kiểu GraphRAGStore
    llm: LLM  # Định nghĩa biến llm kiểu LLM

    def custom_query(self, query_str: str) -> str:  # Định nghĩa phương thức custom_query
        """Process all community summaries to generate answers to a specific query."""  # Xử lý tất cả các bản tóm tắt cộng đồng để tạo câu trả lời cho một truy vấn cụ thể.
        community_summaries = self.graph_store.get_community_summaries()  # Lấy các bản tóm tắt cộng đồng từ graph_store
        community_answers = [  # Tạo danh sách câu trả lời cộng đồng
            self.generate_answer_from_summary(community_summary, query_str)  # Gọi phương thức generate_answer_from_summary
            for _, community_summary in community_summaries.items()  # Lặp qua các bản tóm tắt cộng đồng
        ]

        final_answer = self.aggregate_answers(community_answers)  # Gọi phương thức aggregate_answers để tạo câu trả lời cuối cùng
        return final_answer  # Trả về câu trả lời cuối cùng

    def generate_answer_from_summary(self, community_summary, query):  # Định nghĩa phương thức generate_answer_from_summary
        """Generate an answer from a community summary based on a given query using LLM."""  # Tạo câu trả lời từ một bản tóm tắt cộng đồng dựa trên truy vấn đã cho sử dụng LLM.
        prompt = (  # Tạo prompt
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [  # Tạo danh sách messages
            ChatMessage(role="system", content=prompt),  # Tạo tin nhắn hệ thống
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",  # Tạo tin nhắn của người dùng
            ),
        ]
        response = self.llm.chat(messages)  # Gửi tin nhắn tới LLM và nhận phản hồi
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()  # Làm sạch phản hồi nhận được
        return cleaned_response  # Trả về phản hồi đã làm sạch

    def aggregate_answers(self, community_answers):  # Định nghĩa phương thức aggregate_answers
        """Aggregate individual community answers into a final, coherent response."""  # Tổng hợp các câu trả lời cộng đồng cá nhân thành một phản hồi cuối cùng, mạch lạc.
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."  # Tạo prompt
        messages = [  # Tạo danh sách messages
            ChatMessage(role="system", content=prompt),  # Tạo tin nhắn hệ thống
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",  # Tạo tin nhắn của người dùng
            ),
        ]
        final_response = self.llm.chat(messages)  # Gửi tin nhắn tới LLM và nhận phản hồi cuối cùng
        cleaned_final_response = re.sub(  # Làm sạch phản hồi cuối cùng
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response  # Trả về phản hồi cuối cùng đã làm sạch


from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)


KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""

entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

def parse_fn(response_str: str) -> Any:
    # Tìm tất cả các entities trong response_str sử dụng entity_pattern
    entities = re.findall(entity_pattern, response_str)

    # Tìm tất cả các relationships trong response_str sử dụng relationship_pattern
    relationships = re.findall(relationship_pattern, response_str)

    # Trả về danh sách các entities và relationships
    return entities, relationships



kg_extractor = GraphRAGExtractor(
    llm=llm,  # Sử dụng LLM đã được định nghĩa trước đó
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,  # Sử dụng prompt template để trích xuất các triplet
    max_paths_per_chunk=2,  # Đặt số lượng đường dẫn tối đa mỗi chunk là 2
    parse_fn=parse_fn,  # Sử dụng hàm parse_fn để phân tích kết quả trả về
)


from llama_index.core import PropertyGraphIndex

index = PropertyGraphIndex(
    nodes=nodes,  # Sử dụng danh sách các nodes đã được định nghĩa trước đó
    property_graph_store=GraphRAGStore(),  # Sử dụng GraphRAGStore để lưu trữ đồ thị
    kg_extractors=[kg_extractor],  # Sử dụng danh sách các kg_extractor để trích xuất tri thức
    show_progress=True,  # Hiển thị tiến trình khi xây dựng index
)
list(index.property_graph_store.graph.nodes.values())[1]
