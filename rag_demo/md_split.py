from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 加载Markdown
# loader = UnstructuredMarkdownLoader("/Users/samlee/Downloads/Untitled3.md")
loader = TextLoader("/Users/samlee/Downloads/Untitled3.md", encoding='utf-8')
docs = loader.load()

# 定义标题分隔符
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# 使用标题分块器
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,
                                      strip_headers=False)
chunks = splitter.split_text(docs[0].page_content)

# chunks 现在是按标题结构切分的语义完整块
for chunk in chunks:
    print(chunk.page_content)
    print("*****")
